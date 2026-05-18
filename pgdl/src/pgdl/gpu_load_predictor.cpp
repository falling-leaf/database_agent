#include "gpu_load_predictor.h"
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

GPULoadPredictor::RuntimeModelImpl::RuntimeModelImpl()
    : hidden_dim_(64) {

    w_fc1 = register_module("w_fc1", torch::nn::Linear(7, 64));
    w_fc2 = register_module("w_fc2", torch::nn::Linear(64, 32));

    r_fc1 = register_module("r_fc1", torch::nn::Linear(5, 32));
    r_fc2 = register_module("r_fc2", torch::nn::Linear(32, 16));

    mlp = register_module("mlp", torch::nn::Linear(32 + 16, 16));
    output = register_module("output", torch::nn::Linear(16, 1));

    InitializeWeights();
}

void GPULoadPredictor::RuntimeModelImpl::InitializeWeights() {
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->bias.defined()) {
                torch::nn::init::zeros_(linear->bias);
            }
        }
    }
}

torch::Tensor GPULoadPredictor::RuntimeModelImpl::forward(
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

GPULoadPredictor::GPULoadPredictor()
    : model_loaded_(false),
      model_(RuntimeModel()),
      workload_mean_(7, 0.0),
      workload_std_(7, 1.0),
      resource_mean_(5, 0.0),
      resource_std_(5, 1.0),
      target_mean_(0.0),
      target_std_(1.0) {}

GPULoadPredictor::~GPULoadPredictor() {}

bool GPULoadPredictor::ExtractTrainingData(
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
        "cuda_cores, gpu_freq, util, mem_used, mem_total, "
        "execution_runtime_us "
        "FROM stanford_dogs_gpu_load_training_data "
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

        std::vector<double> r(5);
        for (int j = 0; j < 5; ++j) {
            r[j] = DatumGetFloat8(SPI_getbinval(tuple, desc, 8 + j, &isnull));
            if (std::isnan(r[j]) || std::isinf(r[j])) {
                elog(WARNING, "Invalid resource value at row %lu, col %d", i, j);
                r[j] = 0.0;
            }
        }

        double runtime =
            DatumGetInt32(SPI_getbinval(tuple, desc, 13, &isnull));

        if (std::isnan(runtime) || std::isinf(runtime) || runtime <= 0) {
            elog(WARNING, "Invalid runtime at row %lu: %f", i, runtime);
            continue;
        }

        workloads.push_back(w);
        resources.push_back(r);
        targets.push_back(runtime);
        
        if (runtime < min_runtime) min_runtime = runtime;
        if (runtime > max_runtime) max_runtime = runtime;
        sum_runtime += runtime;
    }

    SPI_finish();

    if (targets.empty()) {
        elog(ERROR, "No valid GPU training data found");
        return false;
    }

    elog(INFO, "Extracted %zu valid GPU training samples", targets.size());
    elog(INFO, "Runtime stats: min=%.6f, max=%.6f, mean=%.6f", 
         min_runtime, max_runtime, sum_runtime / targets.size());
    return true;
}

void GPULoadPredictor::ComputeNormalizationParams(
    const std::vector<std::vector<double>>& workloads,
    const std::vector<std::vector<double>>& resources,
    const std::vector<double>& targets
) {
    size_t N = targets.size();

    double min_target = DBL_MAX;
    double max_target = 0.0;
    
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

    for (size_t j = 0; j < 5; ++j) {
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
    for (double v : targets) {
        sum += v;
        sum_sq += v * v;
        if (v < min_target) min_target = v;
        if (v > max_target) max_target = v;
    }
    target_mean_ = sum / N;
    double variance = (sum_sq / N) - (target_mean_ * target_mean_);
    target_std_ = std::sqrt(std::max(variance, 1e-8));
}

torch::Tensor GPULoadPredictor::NormalizeWorkload(
    const std::vector<std::vector<double>>& data
) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 7}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < 7; ++j)
            accessor[i][j] = (data[i][j] - workload_mean_[j]) / workload_std_[j];

    return tensor;
}

torch::Tensor GPULoadPredictor::NormalizeResource(
    const std::vector<std::vector<double>>& data
) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 5}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < 5; ++j)
            accessor[i][j] = (data[i][j] - resource_mean_[j]) / resource_std_[j];

    return tensor;
}

torch::Tensor GPULoadPredictor::NormalizeTarget(
    const std::vector<double>& data
) {
    size_t N = data.size();
    auto tensor = torch::zeros({(int64_t)N, 1}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        accessor[i][0] = (data[i] - target_mean_) / target_std_;

    return tensor;
}

bool GPULoadPredictor::TrainModel() {
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

    torch::optim::Adam opt(model_->parameters(),
        torch::optim::AdamOptions(1e-3));

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
            elog(INFO, "GPU Epoch %d | Loss %.6f",
                 epoch, total_loss.item<double>());
        }

        if (loss_val < 1e-8) {
            elog(INFO, "GPU Converged at epoch %d", epoch);
            converged = true;
            final_epoch = epoch;
            break;
        }
    }

    if (!converged) {
        elog(WARNING, "GPU Training did not fully converge, final loss may be high");
    }

    model_loaded_ = true;
    return true;
}

double GPULoadPredictor::Predict(
    const std::vector<double>& workload,
    const std::vector<double>& gpu_resource
) {
    if (!model_loaded_)
        return 0.0;

    torch::NoGradGuard guard;
    model_->eval();

    std::vector<float> w_norm(7), r_norm(5);
    for (int i = 0; i < 7; ++i)
        w_norm[i] = (workload[i] - workload_mean_[i]) / workload_std_[i];
    for (int i = 0; i < 5; ++i)
        r_norm[i] = (gpu_resource[i] - resource_mean_[i]) / resource_std_[i];

    auto W = torch::tensor(w_norm, torch::kFloat32).unsqueeze(0);
    auto R = torch::tensor(r_norm, torch::kFloat32).unsqueeze(0);

    auto y_norm = model_->forward(W, R);
    double y = y_norm.item<double>() * target_std_ + target_mean_;

    return std::max(0.0, y);
}

bool GPULoadPredictor::SaveModel(const std::string& model_path) {
    if (!model_loaded_)
        return false;

    torch::serialize::OutputArchive ar;
    ar.write("workload_mean", torch::tensor(workload_mean_));
    ar.write("workload_std", torch::tensor(workload_std_));
    ar.write("resource_mean", torch::tensor(resource_mean_));
    ar.write("resource_std", torch::tensor(resource_std_));
    ar.write("target_mean", torch::tensor(target_mean_));
    ar.write("target_std", torch::tensor(target_std_));

    for (auto& p : model_->named_parameters(true))
        ar.write(p.key(), p.value());

    for (auto& b : model_->named_buffers(true))
        ar.write(b.key(), b.value());

    ar.save_to(model_path);
    return true;
}

bool GPULoadPredictor::LoadModel(const std::string& model_path) {
    try {
        model_ = RuntimeModel();

        torch::serialize::InputArchive ar;
        ar.load_from(model_path);

        torch::Tensor wm, ws, rm, rs, tm, ts;
        ar.read("workload_mean", wm);
        ar.read("workload_std", ws);
        ar.read("resource_mean", rm);
        ar.read("resource_std", rs);
        ar.read("target_mean", tm);
        ar.read("target_std", ts);

        auto wm_acc = wm.accessor<float, 1>();
        auto ws_acc = ws.accessor<float, 1>();
        auto rm_acc = rm.accessor<float, 1>();
        auto rs_acc = rs.accessor<float, 1>();

        for (int i = 0; i < 7; ++i) {
            workload_mean_[i] = wm_acc[i];
            workload_std_[i] = ws_acc[i];
        }
        for (int i = 0; i < 5; ++i) {
            resource_mean_[i] = rm_acc[i];
            resource_std_[i] = rs_acc[i];
        }
        target_mean_ = tm.item<double>();
        target_std_ = ts.item<double>();

        for (auto& p : model_->named_parameters(true)) {
            torch::Tensor t;
            ar.read(p.key(), t);
            p.value().data().copy_(t);
        }

        auto named_buffers = model_->named_buffers(true);
        for (const auto& pair : named_buffers) {
            torch::Tensor tensor;
            if (ar.try_read(pair.key(), tensor)) {
                pair.value().data().copy_(tensor);
            }
        }

        model_->eval();
        model_loaded_ = true;
        return true;

    } catch (const torch::Error& e) {
        elog(INFO, "GPULoadPredictor::LoadModel - Error loading model: %s", e.what());
        return false;
    }
}
