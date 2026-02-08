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

    // 预测 execution_runtime_us
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
    
    // 辅助函数
    void ComputeNormalizationParams(
        const std::vector<std::vector<double>>& workloads,
        const std::vector<std::vector<double>>& resources,
        const std::vector<double>& targets
    );
    
    torch::Tensor NormalizeWorkload(const std::vector<std::vector<double>>& data);
    torch::Tensor NormalizeResource(const std::vector<std::vector<double>>& data);
    torch::Tensor NormalizeTarget(const std::vector<double>& data);

    /* ================= Runtime Model ================= */

    class RuntimeModelImpl : public torch::nn::Module {
    public:
        RuntimeModelImpl();
        void InitializeWeights();

        // prev_state: [1, batch, hidden]
        torch::Tensor forward(
            torch::Tensor workload_x,
            torch::Tensor resource_x,
            torch::Tensor prev_state,
            torch::Tensor& next_state
        );

        int hidden_dim() const { return hidden_dim_; }

    private:
        // workload encoder
        torch::nn::Linear w_fc1{nullptr};
        torch::nn::Linear w_fc2{nullptr};

        // PREACT state transition
        torch::nn::GRU gru{nullptr};

        // runtime head
        torch::nn::Linear y_fc1{nullptr};
        torch::nn::Linear y_fc2{nullptr};

        torch::nn::Dropout dropout{nullptr};

        int hidden_dim_;
    };

    TORCH_MODULE(RuntimeModel);
    RuntimeModel model_;
};

#endif  // _CPU_LOAD_PREDICTOR_H_
