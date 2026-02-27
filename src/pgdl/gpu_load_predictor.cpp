#include "gpu_load_predictor.h"
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

GPULoadPredictor::RuntimeModelImpl::RuntimeModelImpl()
    : hidden_dim_(32) {

    // workload: 7 dims
    w_fc1 = register_module("w_fc1", torch::nn::Linear(7, 32));
    w_fc2 = register_module("w_fc2", torch::nn::Linear(32, 16));

    // GRU: (workload_embed + gpu_resource(5))
    gru = register_module(
        "gru",
        torch::nn::GRU(
            torch::nn::GRUOptions(16 + 5, hidden_dim_).batch_first(true))
    );

    // runtime head
    y_fc1 = register_module(
        "y_fc1",
        torch::nn::Linear(hidden_dim_ + 16 + 5, 16)
    );
    y_fc2 = register_module("y_fc2", torch::nn::Linear(16, 1));

    dropout = register_module("dropout", torch::nn::Dropout(0.1));

    InitializeWeights();
}

void GPULoadPredictor::RuntimeModelImpl::InitializeWeights() {
    for (auto& module : modules(false)) {
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
    auto state_seq = std::get<0>(gru_out);   // [B,1,H]
    next_state = std::get<1>(gru_out);       // [1,B,H]
    auto state_t = state_seq.squeeze(1);     // [B,H]

    // ---- runtime prediction ----
    auto y_in = torch::cat({state_t, w, resource_x}, 1);
    auto y = torch::relu(y_fc1(y_in));
    y = dropout(y);
    return y_fc2(y);
}

/* ================= GPULoadPredictor ================= */

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

/* ---------- ExtractTrainingData ---------- */

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
        "FROM gpu_load_training_data "
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
            DatumGetFloat8(SPI_getbinval(tuple, desc, 13, &isnull));

        if (std::isnan(runtime) || std::isinf(runtime) || runtime <= 0) {
            elog(WARNING, "Invalid runtime at row %lu: %f", i, runtime);
            continue;
        }

        workloads.push_back(w);
        resources.push_back(r);
        targets.push_back(runtime);
    }

    SPI_finish();

    if (targets.empty()) {
        elog(ERROR, "No valid GPU training data found");
        return false;
    }

    elog(INFO, "Extracted %zu valid GPU training samples", targets.size());
    return true;
}

/* ---------- Normalization ---------- */

void GPULoadPredictor::ComputeNormalizationParams(
    const std::vector<std::vector<double>>& workloads,
    const std::vector<std::vector<double>>& resources,
    const std::vector<double>& targets
) {
    size_t N = targets.size();

    for (size_t j = 0; j < 7; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += workloads[i][j];
            sum_sq += workloads[i][j] * workloads[i][j];
        }
        workload_mean_[j] = sum / N;
        double var = sum_sq / N - workload_mean_[j] * workload_mean_[j];
        workload_std_[j] = std::sqrt(std::max(var, 1e-8));
    }

    for (size_t j = 0; j < 5; ++j) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < N; ++i) {
            sum += resources[i][j];
            sum_sq += resources[i][j] * resources[i][j];
        }
        resource_mean_[j] = sum / N;
        double var = sum_sq / N - resource_mean_[j] * resource_mean_[j];
        resource_std_[j] = std::sqrt(std::max(var, 1e-8));
    }

    double sum = 0.0, sum_sq = 0.0;
    for (double v : targets) {
        sum += v;
        sum_sq += v * v;
    }
    target_mean_ = sum / N;
    double var = sum_sq / N - target_mean_ * target_mean_;
    target_std_ = std::sqrt(std::max(var, 1e-8));
}

torch::Tensor GPULoadPredictor::NormalizeWorkload(
    const std::vector<std::vector<double>>& data
) {
    size_t N = data.size();
    auto t = torch::zeros({(int64_t)N, 7}, torch::kFloat32);
    auto acc = t.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < 7; ++j)
            acc[i][j] = (data[i][j] - workload_mean_[j]) / workload_std_[j];

    return t;
}

torch::Tensor GPULoadPredictor::NormalizeResource(
    const std::vector<std::vector<double>>& data
) {
    size_t N = data.size();
    auto t = torch::zeros({(int64_t)N, 5}, torch::kFloat32);
    auto acc = t.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < 5; ++j)
            acc[i][j] = (data[i][j] - resource_mean_[j]) / resource_std_[j];

    return t;
}

torch::Tensor GPULoadPredictor::NormalizeTarget(
    const std::vector<double>& data
) {
    size_t N = data.size();
    auto t = torch::zeros({(int64_t)N, 1}, torch::kFloat32);
    auto acc = t.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i)
        acc[i][0] = (data[i] - target_mean_) / target_std_;

    return t;
}

/* ---------- TrainModel ---------- */

bool GPULoadPredictor::TrainModel() {
    std::vector<std::vector<double>> workloads, resources;
    std::vector<double> targets;

    if (!ExtractTrainingData(workloads, resources, targets))
        return false;

    ComputeNormalizationParams(workloads, resources, targets);

    auto W = NormalizeWorkload(workloads);
    auto R = NormalizeResource(resources);
    auto Y = NormalizeTarget(targets);

    model_ = RuntimeModel();
    model_->train();

    torch::optim::Adam opt(model_->parameters(),
        torch::optim::AdamOptions(1e-4));

    int64_t N = targets.size();

    for (int epoch = 0; epoch < 100; ++epoch) {
        opt.zero_grad();
        auto total_loss = torch::zeros({});

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

        total_loss = total_loss / N;

        if (torch::isnan(total_loss).item<bool>())
            return false;

        total_loss.backward();
        torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
        opt.step();

        if (epoch % 10 == 0 || epoch == 99) {
            elog(INFO, "GPU Epoch %d | Loss %.6f",
                 epoch, total_loss.item<double>());
        }
    }

    model_loaded_ = true;
    return true;
}

/* ---------- Predict ---------- */

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

    auto W = torch::tensor(w_norm).unsqueeze(0);
    auto R = torch::tensor(r_norm).unsqueeze(0);

    auto state = torch::zeros({1, 1, model_->hidden_dim()});
    torch::Tensor next_state;

    auto y_norm = model_->forward(W, R, state, next_state);
    double y = y_norm.item<double>() * target_std_ + target_mean_;

    return std::max(0.0, y);
}

/* ---------- Save / Load ---------- */

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

        torch::NoGradGuard no_grad;  // ⭐ 关键

        torch::serialize::InputArchive ar;
        ar.load_from(model_path);

        torch::Tensor wm, ws, rm, rs, tm, ts;
        ar.read("workload_mean", wm);
        ar.read("workload_std", ws);
        ar.read("resource_mean", rm);
        ar.read("resource_std", rs);
        ar.read("target_mean", tm);
        ar.read("target_std", ts);

        for (int i = 0; i < 7; ++i) {
            workload_mean_[i] = wm[i].item<double>();
            workload_std_[i] = ws[i].item<double>();
        }
        for (int i = 0; i < 5; ++i) {
            resource_mean_[i] = rm[i].item<double>();
            resource_std_[i] = rs[i].item<double>();
        }
        target_mean_ = tm.item<double>();
        target_std_ = ts.item<double>();

        for (auto& p : model_->named_parameters(true)) {
            torch::Tensor t;
            ar.read(p.key(), t);
            p.value().copy_(t);  // ✅ NoGradGuard 下合法
        }

        model_loaded_ = true;
        model_->eval();
        return true;

    } catch (const torch::Error& e) {
        elog(INFO, "GPULoadPredictor::LoadModel - Error loading model: %s", e.what());
        return false;
    }
}

