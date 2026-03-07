#include "cpu_load_predictor.h"
#include <algorithm>
#include <cmath>
#include <cfloat>

extern "C" {
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/memutils.h"
#include "miscadmin.h"
}

CPULoadPredictor::RuntimeModelImpl::RuntimeModelImpl()
    : hidden_dim_(64) {

    w_fc1 = register_module("w_fc1", torch::nn::Linear(7, 64));
    w_fc2 = register_module("w_fc2", torch::nn::Linear(64, 32));

    r_fc1 = register_module("r_fc1", torch::nn::Linear(2, 32));
    r_fc2 = register_module("r_fc2", torch::nn::Linear(32, 16));

    mlp = register_module("mlp", torch::nn::Linear(32 + 16, 16));
    output = register_module("output", torch::nn::Linear(16, 1));

    InitializeWeights();
}

void CPULoadPredictor::RuntimeModelImpl::InitializeWeights() {
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->bias.defined()) {
                torch::nn::init::zeros_(linear->bias);
            }
        }
    }
}

torch::Tensor CPULoadPredictor::RuntimeModelImpl::forward(
    torch::Tensor workload_x,
    torch::Tensor resource_x
) {
    auto w = torch::relu(w_fc1(workload_x));
    w = torch::relu(w_fc2(w));

    auto r = torch::relu(r_fc1(resource_x));
    r = torch::relu(r_fc2(r));

    auto combined = torch::cat({w, r}, 1);
    auto y = torch::relu(mlp(combined));
    return output(y);
}

CPULoadPredictor::CPULoadPredictor()
    : model_loaded_(false),
      model_(RuntimeModel()),
      workload_mean_(7, 0.0),
      workload_std_(7, 1.0),
      resource_mean_(2, 0.0),
      resource_std_(2, 1.0),
      target_mean_(0.0),
      target_std_(1.0) {}

CPULoadPredictor::~CPULoadPredictor() {}

bool CPULoadPredictor::ExtractTrainingData(
    std::vector<std::vector<double>>& workloads,
    std::vector<std::vector<double>>& resources,
    std::vector<double>& targets
) {
    if (SPI_connect() != SPI_OK_CONNECT)
        return false;

    const char* sql =
        "SELECT "
        "data_shape_1, data_shape_2, data_shape_3, data_shape_4, "
        "model_mac_count, model_param_count, model_param_size, "
        "cpu_load, cpu_cores, execution_runtime_us "
        "FROM stanford_dogs_cpu_load_training_data "
        "ORDER BY id";

    if (SPI_execute(sql, true, 0) != SPI_OK_SELECT) {
        SPI_finish();
        return false;
    }

    workloads.clear();
    resources.clear();
    targets.clear();

    double min_runtime = DBL_MAX;
    double max_runtime = 0.0;
    double sum_runtime = 0.0;
    int skipped_count = 0;
    int total_count = 0;

    for (uint64 i = 0; i < SPI_processed; ++i) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc desc = SPI_tuptable->tupdesc;
        bool isnull;

        std::vector<double> w(7);
        for (int j = 0; j < 7; ++j) {
            w[j] = DatumGetInt32(SPI_getbinval(tuple, desc, j + 1, &isnull));
            if (std::isnan(w[j]) || std::isinf(w[j])) {
                elog(WARNING, "Invalid workload value at row %lu, col %d", i, j);
                w[j] = 0.0;
            }
        }

        double cpu_load = DatumGetFloat8(SPI_getbinval(tuple, desc, 8, &isnull));
        int cpu_cores = DatumGetInt32(SPI_getbinval(tuple, desc, 9, &isnull));
        double runtime = DatumGetInt32(SPI_getbinval(tuple, desc, 10, &isnull));
        // elog(INFO, "workload: %f, %f, %f, %f, %f, %f, %f, cpu_load: %f, cpu_cores: %d, runtime: %f", 
        //      w[0], w[1], w[2], w[3], w[4], w[5], w[6], cpu_load, cpu_cores, runtime);

        total_count++;
        
        if (std::isnan(cpu_load) || std::isinf(cpu_load)) cpu_load = 0.5;
        
        if (std::isnan(runtime) || std::isinf(runtime)) {
            elog(WARNING, "Row %lu: runtime is NaN or Inf (%f), skipping", i, runtime);
            skipped_count++;
            continue;
        }
        
        if (runtime == 0) {
            elog(WARNING, "Row %lu: runtime is exactly 0, this is suspicious! cpu_load=%f, cpu_cores=%d", 
                 i, cpu_load, cpu_cores);
            skipped_count++;
            continue;
        }
        
        if (runtime < 0) {
            elog(WARNING, "Row %lu: runtime is negative (%f), skipping", i, runtime);
            skipped_count++;
            continue;
        }

        workloads.push_back(w);
        resources.push_back({cpu_load, (double)cpu_cores});
        targets.push_back(runtime);
        
        if (runtime < min_runtime) min_runtime = runtime;
        if (runtime > max_runtime) max_runtime = runtime;
        sum_runtime += runtime;
    }

    SPI_finish();
    
    elog(INFO, "Total rows in table: %d, Skipped: %d, Valid: %zu", 
         total_count, skipped_count, targets.size());
    
    if (targets.empty()) {
        elog(ERROR, "No valid training data found after filtering");
        return false;
    }
    
    elog(INFO, "Runtime stats: min=%.6f, max=%.6f, mean=%.6f", 
         min_runtime, max_runtime, sum_runtime / targets.size());
    return true;
}

void CPULoadPredictor::ComputeNormalizationParams(
    const std::vector<std::vector<double>>& workloads,
    const std::vector<std::vector<double>>& resources,
    const std::vector<double>& targets
) {
    size_t N = targets.size();
    elog(INFO, "target sample: %f", targets[0]);
    
    double min_target = DBL_MAX;
    double max_target = 0.0;
    double sum_target = 0.0;
    
    for (size_t j = 0; j < 7; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += workloads[i][j];
            sum_sq += workloads[i][j] * workloads[i][j];
        }
        workload_mean_[j] = sum / N;
        double variance = (sum_sq / N) - (workload_mean_[j] * workload_mean_[j]);
        workload_std_[j] = std::sqrt(std::max(variance, 1e-8));
    }
    
    for (size_t j = 0; j < 2; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += resources[i][j];
            sum_sq += resources[i][j] * resources[i][j];
        }
        resource_mean_[j] = sum / N;
        double variance = (sum_sq / N) - (resource_mean_[j] * resource_mean_[j]);
        resource_std_[j] = std::sqrt(std::max(variance, 1e-8));
    }
    
    double sum = 0.0, sum_sq = 0.0;
    for (double val : targets) {
        sum += val;
        sum_sq += val * val;
        if (val < min_target) min_target = val;
        if (val > max_target) max_target = val;
    }
    target_mean_ = sum / N;
    double variance = (sum_sq / N) - (target_mean_ * target_mean_);
    target_std_ = std::sqrt(std::max(variance, 1e-8));
    
    elog(INFO, "Normalization params computed:");
    elog(INFO, "  Target: min=%.6f, max=%.6f, mean=%.6f, std=%.6f", 
         min_target, max_target, target_mean_, target_std_);
    elog(INFO, "  Workload[0]: mean=%.6f, std=%.6f", workload_mean_[0], workload_std_[0]);
}

torch::Tensor CPULoadPredictor::NormalizeWorkload(const std::vector<std::vector<double>>& data) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 7}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 7; ++j) {
            accessor[i][j] = (data[i][j] - workload_mean_[j]) / workload_std_[j];
        }
    }
    return tensor;
}

torch::Tensor CPULoadPredictor::NormalizeResource(const std::vector<std::vector<double>>& data) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 2}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            accessor[i][j] = (data[i][j] - resource_mean_[j]) / resource_std_[j];
        }
    }
    return tensor;
}

torch::Tensor CPULoadPredictor::NormalizeTarget(const std::vector<double>& data) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 1}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    
    for (size_t i = 0; i < N; ++i) {
        accessor[i][0] = (data[i] - target_mean_) / target_std_;
    }
    return tensor;
}

bool CPULoadPredictor::TrainModel() {
    std::vector<std::vector<double>> workloads, resources;
    std::vector<double> targets;

    if (!ExtractTrainingData(workloads, resources, targets))
        return false;

    int64_t N = targets.size();
    
    ComputeNormalizationParams(workloads, resources, targets);
    
    auto W = NormalizeWorkload(workloads);
    auto R = NormalizeResource(resources);
    auto Y = NormalizeTarget(targets);

    if (torch::isnan(W).any().item<bool>() || torch::isnan(R).any().item<bool>() || torch::isnan(Y).any().item<bool>()) {
        elog(ERROR, "NaN detected in normalized data");
        return false;
    }

    model_ = RuntimeModel();
    model_->train();
    
    torch::optim::Adam opt(model_->parameters(), torch::optim::AdamOptions(1e-3));

    bool converged = false;
    int final_epoch = 0;
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        opt.zero_grad();
        auto total_loss = torch::zeros({});

        for (int64_t i = 0; i < N; ++i) {
            auto pred = model_->forward(W[i].unsqueeze(0), R[i].unsqueeze(0));
            total_loss += torch::mse_loss(pred, Y[i].unsqueeze(0));
        }

        total_loss = total_loss / N;
        
        if (torch::isnan(total_loss).item<bool>() || torch::isinf(total_loss).item<bool>()) {
            elog(ERROR, "NaN/Inf loss at epoch %d, stopping training", epoch);
            return false;
        }

        total_loss.backward();
        
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        
        opt.step();

        double loss_val = total_loss.item<double>();
        
        if (epoch % 20 == 0 || epoch == 99) {
            elog(INFO, "Epoch %d | Loss %.6f", epoch, loss_val);
        }
        
        if (loss_val < 1e-8) {
            elog(INFO, "Converged at epoch %d", epoch);
            converged = true;
            final_epoch = epoch;
            break;
        }
    }

    if (!converged) {
        elog(WARNING, "Training did not fully converge, final loss may be high");
    }

    model_loaded_ = true;
    return true;
}

double CPULoadPredictor::Predict(
    const std::vector<double>& workload,
    double cpu_load,
    int cpu_cores
) {
    if (!model_loaded_)
        return 0.0;

    torch::NoGradGuard guard;
    model_->eval();

    std::vector<float> w_norm(7);
    for (size_t i = 0; i < 7; ++i) {
        w_norm[i] = (workload[i] - workload_mean_[i]) / workload_std_[i];
    }
    
    std::vector<float> r_norm(2);
    r_norm[0] = (cpu_load - resource_mean_[0]) / resource_std_[0];
    r_norm[1] = ((double)cpu_cores - resource_mean_[1]) / resource_std_[1];

    auto W = torch::tensor(w_norm, torch::kFloat32).unsqueeze(0);
    auto R = torch::tensor(r_norm, torch::kFloat32).unsqueeze(0);

    auto y_norm = model_->forward(W, R);
    
    double y = y_norm.item<double>() * target_std_ + target_mean_;

    return std::max(0.0, y);
}

bool CPULoadPredictor::SaveModel(const std::string& model_path) {
    if (!model_loaded_) {
        elog(WARNING, "CPULoadPredictor: Cannot save model - model not trained");
        return false;
    }

    try {
        torch::serialize::OutputArchive archive;
        
        archive.write("workload_mean", torch::tensor(workload_mean_));
        archive.write("workload_std", torch::tensor(workload_std_));
        archive.write("resource_mean", torch::tensor(resource_mean_));
        archive.write("resource_std", torch::tensor(resource_std_));
        archive.write("target_mean", torch::tensor(target_mean_));
        archive.write("target_std", torch::tensor(target_std_));
        
        auto named_params = model_->named_parameters(/*recurse=*/true);
        for (const auto& pair : named_params) {
            archive.write(pair.key(), pair.value());
        }
        
        auto named_buffers = model_->named_buffers(/*recurse=*/true);
        for (const auto& pair : named_buffers) {
            archive.write(pair.key(), pair.value());
        }
        
        archive.save_to(model_path);
        
        elog(INFO, "CPULoadPredictor: Model saved to %s", model_path.c_str());
        return true;
        
    } catch (const c10::Error& e) {
        elog(ERROR, "CPULoadPredictor: Failed to save model - %s", e.what());
        return false;
    } catch (const std::exception& e) {
        elog(ERROR, "CPULoadPredictor: Failed to save model - %s", e.what());
        return false;
    }
}

bool CPULoadPredictor::LoadModel(const std::string& model_path) {
    try {
        model_ = RuntimeModel();
        
        torch::serialize::InputArchive archive;
        archive.load_from(model_path);
        
        torch::Tensor w_mean, w_std, r_mean, r_std, t_mean, t_std;
        archive.read("workload_mean", w_mean);
        archive.read("workload_std", w_std);
        archive.read("resource_mean", r_mean);
        archive.read("resource_std", r_std);
        archive.read("target_mean", t_mean);
        archive.read("target_std", t_std);
        
        auto w_mean_acc = w_mean.accessor<float, 1>();
        auto w_std_acc = w_std.accessor<float, 1>();
        auto r_mean_acc = r_mean.accessor<float, 1>();
        auto r_std_acc = r_std.accessor<float, 1>();
        
        for (size_t i = 0; i < 7; ++i) {
            workload_mean_[i] = w_mean_acc[i];
            workload_std_[i] = w_std_acc[i];
        }
        for (size_t i = 0; i < 2; ++i) {
            resource_mean_[i] = r_mean_acc[i];
            resource_std_[i] = r_std_acc[i];
        }
        target_mean_ = t_mean.item<double>();
        target_std_ = t_std.item<double>();
        
        auto named_params = model_->named_parameters(/*recurse=*/true);
        for (const auto& pair : named_params) {
            torch::Tensor tensor;
            archive.read(pair.key(), tensor);
            pair.value().data().copy_(tensor);
        }
        
        auto named_buffers = model_->named_buffers(/*recurse=*/true);
        for (const auto& pair : named_buffers) {
            torch::Tensor tensor;
            if (archive.try_read(pair.key(), tensor)) {
                pair.value().data().copy_(tensor);
            }
        }
        
        model_->eval();
        model_loaded_ = true;
        
        elog(INFO, "CPULoadPredictor: Model loaded from %s", model_path.c_str());
        return true;
        
    } catch (const c10::Error& e) {
        model_loaded_ = false;
        return false;
    } catch (const std::exception& e) {
        model_loaded_ = false;
        return false;
    }
}
