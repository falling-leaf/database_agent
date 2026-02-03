#pragma once
#ifndef _CPU_LOAD_PREDICTOR_H_
#define _CPU_LOAD_PREDICTOR_H_

#include <torch/torch.h>
#include <vector>
#include <string>

extern "C" {
#include "postgres.h"
}

class CPULoadPredictor {
public:
    CPULoadPredictor();
    ~CPULoadPredictor();

    bool TrainModel();
    bool LoadModel(const std::string& model_path);
    bool SaveModel(const std::string& model_path);
    double Predict(double cpu_load, int cpu_cores);

    // ⚠️ 声明与 cpp 中实现必须 100% 一致
    bool ExtractTrainingData(
        std::vector<std::vector<double>>& features,
        std::vector<double>& targets
    );

private:
    bool model_loaded_;

    class SimpleMLPImpl : public torch::nn::Module {
    public:
        SimpleMLPImpl();
        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Linear fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
        torch::nn::Linear fc3{nullptr};
        torch::nn::Dropout dropout{nullptr};
    };

    TORCH_MODULE(SimpleMLP);
    SimpleMLP model_;
};

#endif  // _CPU_LOAD_PREDICTOR_H_
