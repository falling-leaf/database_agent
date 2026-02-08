#include "cpu_load_predictor.h"
#include <algorithm>
#include <cmath>

extern "C" {
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/memutils.h"
#include "miscadmin.h"
}

/* ================= RuntimeModel ================= */

CPULoadPredictor::RuntimeModelImpl::RuntimeModelImpl()
    : hidden_dim_(32) {

    // workload: 7 dims
    w_fc1 = register_module("w_fc1", torch::nn::Linear(7, 32));
    w_fc2 = register_module("w_fc2", torch::nn::Linear(32, 16));

    // GRU: (workload_embed + resource) -> state
    gru = register_module(
        "gru",
        torch::nn::GRU(torch::nn::GRUOptions(16 + 2, hidden_dim_).batch_first(true))
    );

    // runtime head
    y_fc1 = register_module("y_fc1",
        torch::nn::Linear(hidden_dim_ + 16 + 2, 16));
    y_fc2 = register_module("y_fc2", torch::nn::Linear(16, 1));

    dropout = register_module("dropout", torch::nn::Dropout(0.1));
    
    // 初始化权重
    InitializeWeights();
}

void CPULoadPredictor::RuntimeModelImpl::InitializeWeights() {
    // Xavier/Glorot 初始化
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
    torch::Tensor resource_x,
    torch::Tensor prev_state,
    torch::Tensor& next_state
) {
    // ---- workload encoder ----
    auto w = torch::relu(w_fc1(workload_x));
    w = dropout(w);
    w = torch::relu(w_fc2(w));   // [B,16]

    // ---- state transition ----
    auto gru_input = torch::cat({w, resource_x}, 1).unsqueeze(1);
    auto gru_out = gru->forward(gru_input, prev_state);
    auto state_seq = std::get<0>(gru_out);      // [B,1,H]
    next_state = std::get<1>(gru_out);          // [1,B,H]
    auto state_t = state_seq.squeeze(1);        // [B,H]

    // ---- runtime prediction ----
    auto y_in = torch::cat({state_t, w, resource_x}, 1);
    auto y = torch::relu(y_fc1(y_in));
    y = dropout(y);
    return y_fc2(y);
}

/* ================= CPULoadPredictor ================= */

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
        "FROM cpu_load_training_data "
        "ORDER BY id";

    if (SPI_execute(sql, true, 0) != SPI_OK_SELECT) {
        SPI_finish();
        return false;
    }

    workloads.clear();
    resources.clear();
    targets.clear();

    for (uint64 i = 0; i < SPI_processed; ++i) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc desc = SPI_tuptable->tupdesc;
        bool isnull;

        std::vector<double> w(7);
        for (int j = 0; j < 7; ++j) {
            w[j] = DatumGetFloat8(SPI_getbinval(tuple, desc, j + 1, &isnull));
            // 检查 NaN/Inf
            if (std::isnan(w[j]) || std::isinf(w[j])) {
                elog(WARNING, "Invalid workload value at row %lu, col %d", i, j);
                w[j] = 0.0;
            }
        }

        double cpu_load = DatumGetFloat8(SPI_getbinval(tuple, desc, 8, &isnull));
        int cpu_cores = DatumGetInt32(SPI_getbinval(tuple, desc, 9, &isnull));
        double runtime = DatumGetFloat8(SPI_getbinval(tuple, desc, 10, &isnull));

        // 检查有效性
        if (std::isnan(cpu_load) || std::isinf(cpu_load)) cpu_load = 0.5;
        if (std::isnan(runtime) || std::isinf(runtime) || runtime <= 0) {
            elog(WARNING, "Invalid runtime at row %lu: %f", i, runtime);
            continue;  // 跳过无效样本
        }

        workloads.push_back(w);
        resources.push_back({cpu_load, (double)cpu_cores});
        targets.push_back(runtime);
    }

    SPI_finish();
    
    if (targets.empty()) {
        elog(ERROR, "No valid training data found");
        return false;
    }
    
    elog(INFO, "Extracted %zu valid training samples", targets.size());
    return true;
}

void CPULoadPredictor::ComputeNormalizationParams(
    const std::vector<std::vector<double>>& workloads,
    const std::vector<std::vector<double>>& resources,
    const std::vector<double>& targets
) {
    size_t N = targets.size();
    
    // 计算 workload 的均值和标准差
    for (size_t j = 0; j < 7; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += workloads[i][j];
            sum_sq += workloads[i][j] * workloads[i][j];
        }
        workload_mean_[j] = sum / N;
        double variance = (sum_sq / N) - (workload_mean_[j] * workload_mean_[j]);
        workload_std_[j] = std::sqrt(std::max(variance, 1e-8));  // 防止除零
    }
    
    // 计算 resource 的均值和标准差
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
    
    // 计算 target 的均值和标准差
    double sum = 0.0, sum_sq = 0.0;
    for (double val : targets) {
        sum += val;
        sum_sq += val * val;
    }
    target_mean_ = sum / N;
    double variance = (sum_sq / N) - (target_mean_ * target_mean_);
    target_std_ = std::sqrt(std::max(variance, 1e-8));
    
    elog(INFO, "Normalization params computed:");
    elog(INFO, "  Target: mean=%.2f, std=%.2f", target_mean_, target_std_);
    elog(INFO, "  Workload[0]: mean=%.2f, std=%.2f", workload_mean_[0], workload_std_[0]);
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
    
    // 计算归一化参数
    ComputeNormalizationParams(workloads, resources, targets);
    
    // 归一化数据
    auto W = NormalizeWorkload(workloads);
    auto R = NormalizeResource(resources);
    auto Y = NormalizeTarget(targets);

    // 检查归一化后的数据
    if (torch::isnan(W).any().item<bool>() || torch::isnan(R).any().item<bool>() || torch::isnan(Y).any().item<bool>()) {
        elog(ERROR, "NaN detected in normalized data");
        return false;
    }

    // 重置模型（每个样本独立，不使用序列状态）
    model_ = RuntimeModel();
    model_->train();
    
    // 使用较小的学习率和梯度裁剪
    torch::optim::Adam opt(model_->parameters(), torch::optim::AdamOptions(1e-4));

    for (int epoch = 0; epoch < 50; ++epoch) {
        opt.zero_grad();
        auto total_loss = torch::zeros({});

        // 每个样本重置状态（视为独立预测）
        for (int64_t i = 0; i < N; ++i) {
            auto state = torch::zeros({1, 1, model_->hidden_dim()});
            
            torch::Tensor next_state;
            auto pred = model_->forward(
                W[i].unsqueeze(0),
                R[i].unsqueeze(0),
                state,
                next_state
            );
            total_loss += torch::mse_loss(pred, Y[i].unsqueeze(0));
        }

        // 平均损失
        total_loss = total_loss / N;
        
        // 检查损失是否有效
        if (torch::isnan(total_loss).item<bool>() || torch::isinf(total_loss).item<bool>()) {
            elog(ERROR, "NaN/Inf loss at epoch %d, stopping training", epoch);
            return false;
        }

        total_loss.backward();
        
        // 梯度裁剪（防止梯度爆炸）
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        
        opt.step();

        double loss_val = total_loss.item<double>();
        
        // 每10个epoch或首尾epoch打印
        if (epoch % 10 == 0 || epoch == 49) {
            elog(INFO, "Epoch %d | Loss %.6f", epoch, loss_val);
        }
        
        // Early stopping（如果损失足够小）
        if (loss_val < 1e-6) {
            elog(INFO, "Converged at epoch %d", epoch);
            break;
        }
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

    // 归一化输入
    std::vector<float> w_norm(7);
    for (size_t i = 0; i < 7; ++i) {
        w_norm[i] = (workload[i] - workload_mean_[i]) / workload_std_[i];
    }
    
    std::vector<float> r_norm(2);
    r_norm[0] = (cpu_load - resource_mean_[0]) / resource_std_[0];
    r_norm[1] = ((double)cpu_cores - resource_mean_[1]) / resource_std_[1];

    auto W = torch::tensor(w_norm, torch::kFloat32).unsqueeze(0);
    auto R = torch::tensor(r_norm, torch::kFloat32).unsqueeze(0);

    // 使用零状态（无状态预测）
    auto state = torch::zeros({1, 1, model_->hidden_dim()});

    torch::Tensor next_state;
    auto y_norm = model_->forward(W, R, state, next_state);
    
    // 反归一化
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
        
        // 保存归一化参数
        archive.write("workload_mean", torch::tensor(workload_mean_));
        archive.write("workload_std", torch::tensor(workload_std_));
        archive.write("resource_mean", torch::tensor(resource_mean_));
        archive.write("resource_std", torch::tensor(resource_std_));
        archive.write("target_mean", torch::tensor(target_mean_));
        archive.write("target_std", torch::tensor(target_std_));
        
        // 保存模型参数
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
        // 重新初始化模型
        model_ = RuntimeModel();
        
        torch::serialize::InputArchive archive;
        archive.load_from(model_path);
        
        // 加载归一化参数
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
        
        // 加载模型参数
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
        // elog(ERROR, "CPULoadPredictor: Failed to load model - %s", e.what());
        model_loaded_ = false;
        return false;
    } catch (const std::exception& e) {
        // elog(ERROR, "CPULoadPredictor: Failed to load model - %s", e.what());
        model_loaded_ = false;
        return false;
    }
}