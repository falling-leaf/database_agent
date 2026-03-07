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

    double Predict(
        const std::vector<double>& workload,
        double cpu_load,
        int cpu_cores
    );

    bool ExtractTrainingData(
        std::vector<std::vector<double>>& workloads,
        std::vector<std::vector<double>>& resources,
        std::vector<double>& targets
    );

private:
    bool model_loaded_;
    std::vector<double> workload_mean_;
    std::vector<double> workload_std_;
    std::vector<double> resource_mean_;
    std::vector<double> resource_std_;
    double target_mean_;
    double target_std_;
    
    void ComputeNormalizationParams(
        const std::vector<std::vector<double>>& workloads,
        const std::vector<std::vector<double>>& resources,
        const std::vector<double>& targets
    );
    
    torch::Tensor NormalizeWorkload(const std::vector<std::vector<double>>& data);
    torch::Tensor NormalizeResource(const std::vector<std::vector<double>>& data);
    torch::Tensor NormalizeTarget(const std::vector<double>& data);

    class RuntimeModelImpl : public torch::nn::Module {
    public:
        RuntimeModelImpl();
        void InitializeWeights();

        torch::Tensor forward(
            torch::Tensor workload_x,
            torch::Tensor resource_x
        );

        int hidden_dim() const { return hidden_dim_; }

    private:
        torch::nn::Linear w_fc1{nullptr};
        torch::nn::Linear w_fc2{nullptr};

        torch::nn::Linear r_fc1{nullptr};
        torch::nn::Linear r_fc2{nullptr};

        torch::nn::Linear mlp{nullptr};
        torch::nn::Linear output{nullptr};

        int hidden_dim_;
    };

    TORCH_MODULE(RuntimeModel);
    RuntimeModel model_;
};

#endif  // _CPU_LOAD_PREDICTOR_H_
