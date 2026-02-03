#include "cpu_load_predictor.h"
#include <algorithm>

extern "C" {
#include "executor/spi.h"
#include "utils/builtins.h"
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/memutils.h"
#include "miscadmin.h"
}

/* ================= SimpleMLP ================= */

CPULoadPredictor::SimpleMLPImpl::SimpleMLPImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(2, 32));
    fc2 = register_module("fc2", torch::nn::Linear(32, 16));
    fc3 = register_module("fc3", torch::nn::Linear(16, 1));
    dropout = register_module("dropout", torch::nn::Dropout(0.1));
}

torch::Tensor CPULoadPredictor::SimpleMLPImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = dropout(x);
    x = torch::relu(fc2(x));
    x = dropout(x);
    return fc3(x);
}

/* ================= CPULoadPredictor ================= */

CPULoadPredictor::CPULoadPredictor()
    : model_loaded_(false),
      model_(SimpleMLP()) {}

CPULoadPredictor::~CPULoadPredictor() {}

bool CPULoadPredictor::LoadModel(const std::string& path) {
    try {
        if (path.empty()) {
            elog(WARNING, "Model path is empty");
            return false;
        }
        if(access(path.c_str(), F_OK) != 0){
            return false;
        }

        torch::load(model_, path);
        model_loaded_ = true;
        elog(INFO, "CPU load model loaded from %s", path.c_str());
        return true;
    } catch (const std::exception& e) {
        elog(ERROR, "Failed to load model from %s: %s", path.c_str(), e.what());
        return false;
    }
}

bool CPULoadPredictor::SaveModel(const std::string& path) {
    if (path.empty()) {
        elog(WARNING, "Model path is empty");
        return false;
    }

    torch::save(model_, path);
    elog(INFO, "CPU load model saved to %s", path.c_str());
    return true;
}

/* ================= 数据抽取（关键函数） ================= */

bool CPULoadPredictor::ExtractTrainingData(
    std::vector<std::vector<double>>& features,
    std::vector<double>& targets
) {
    if (SPI_connect() != SPI_OK_CONNECT) {
        elog(WARNING, "SPI_connect failed");
        return false;
    }

    const char* sql =
        "SELECT cpu_load, cpu_cores, normalized_load_factor "
        "FROM cpu_load_training_data "
        "ORDER BY id";

    int ret = SPI_execute(sql, true, 0);
    if (ret != SPI_OK_SELECT) {
        elog(WARNING, "SPI_execute failed");
        SPI_finish();
        return false;
    }

    features.clear();
    targets.clear();

    for (uint64 i = 0; i < SPI_processed; ++i) {
        HeapTuple tuple = SPI_tuptable->vals[i];
        TupleDesc desc = SPI_tuptable->tupdesc;
        bool isnull;

        double cpu_load =
            DatumGetFloat8(SPI_getbinval(tuple, desc, 1, &isnull));
        int cpu_cores =
            DatumGetInt32(SPI_getbinval(tuple, desc, 2, &isnull));
        double factor =
            DatumGetFloat8(SPI_getbinval(tuple, desc, 3, &isnull));

        features.push_back({cpu_load, static_cast<double>(cpu_cores)});
        targets.push_back(factor);
    }

    SPI_finish();

    elog(INFO, "Extracted %lu training samples", features.size());
    return !features.empty();
}

/* ================= 训练 ================= */

bool CPULoadPredictor::TrainModel() {
    std::vector<std::vector<double>> features;
    std::vector<double> targets;

    if (!ExtractTrainingData(features, targets)) {
        elog(WARNING, "No training data available");
        return false;
    }

    const int64_t N = features.size();

    torch::Tensor X = torch::zeros({N, 2}, torch::kFloat32);
    torch::Tensor y = torch::zeros({N, 1}, torch::kFloat32);

    for (int64_t i = 0; i < N; ++i) {
        X[i][0] = features[i][0];
        X[i][1] = features[i][1];
        y[i][0] = targets[i];
    }

    int64_t val_size = static_cast<int64_t>(0.2 * N);
    int64_t train_size = N - val_size;

    auto X_train = X.narrow(0, val_size, train_size);
    auto y_train = y.narrow(0, val_size, train_size);
    auto X_val   = X.narrow(0, 0, val_size);
    auto y_val   = y.narrow(0, 0, val_size);

    torch::optim::Adam optimizer(model_->parameters(), 1e-3);

    model_->train();
    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.zero_grad();
        auto pred = model_->forward(X_train);
        auto loss = torch::mse_loss(pred, y_train);
        loss.backward();
        optimizer.step();

        if (epoch % 20 == 0 && val_size > 0) {
            model_->eval();
            auto val_loss =
                torch::mse_loss(model_->forward(X_val), y_val);
            model_->train();

            elog(INFO,
                 "Epoch %d | Train %.6f | Val %.6f",
                 epoch,
                 loss.item<double>(),
                 val_loss.item<double>());
        }
    }

    model_loaded_ = true;
    return true;
}

/* ================= 推理 ================= */

double CPULoadPredictor::Predict(double cpu_load, int cpu_cores) {
    if (!model_loaded_) {
        elog(WARNING, "Model not loaded, return default");
        return 1.0;
    }

    torch::NoGradGuard guard;

    torch::Tensor x = torch::tensor(
        {{cpu_load, static_cast<double>(cpu_cores)}},
        torch::kFloat32);

    double y = model_->forward(x).item<double>();
    return std::max(0.1, y);
}
