#include "cpu_load_predictor.h"
#include "spi_connection.h"
#include <algorithm>

extern "C" {
#include "executor/spi.h"
#include "utils/builtins.h"
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
        torch::load(model_, path);
        model_loaded_ = true;
        elog(INFO, "CPU load model loaded from %s", path.c_str());
        return true;
    } catch (...) {
        elog(WARNING, "Failed to load model from %s", path.c_str());
        return false;
    }
}

bool CPULoadPredictor::SaveModel(const std::string& path) {
    try {
        torch::save(model_, path);
        elog(INFO, "CPU load model saved to %s", path.c_str());
        return true;
    } catch (...) {
        elog(WARNING, "Failed to save model to %s", path.c_str());
        return false;
    }
}

bool CPULoadPredictor::TrainModel() {
    std::vector<std::vector<double>> features;
    std::vector<double> targets;

    if (!ExtractTrainingData(features, targets) || features.empty())
        return false;

    const int64_t N = features.size();

    torch::Tensor X = torch::zeros({N, 2});
    torch::Tensor y = torch::zeros({N, 1});

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
            auto val_loss = torch::mse_loss(model_->forward(X_val), y_val);
            model_->train();

            elog(INFO, "Epoch %d | Train %.6f | Val %.6f",
                 epoch,
                 loss.item<double>(),
                 val_loss.item<double>());
        }
    }

    model_loaded_ = true;
    return true;
}

double CPULoadPredictor::Predict(double cpu_load, int cpu_cores) {
    if (!model_loaded_)
        return 1.0;

    torch::NoGradGuard guard;

    torch::Tensor x = torch::tensor(
        {{cpu_load, static_cast<double>(cpu_cores)}},
        torch::kFloat32);

    double y = model_->forward(x).item<double>();
    return std::max(0.1, y);
}
