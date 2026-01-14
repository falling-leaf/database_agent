#pragma once
#ifndef _MODEL_AGENT_H_
#define _MODEL_AGENT_H_

#include "env.h"
#include "model_manager.h"
#include "model_selection.h"
#include "batch_interface.h"
#include "myfunc.h"
#include "vector.h"

#include <stdbool.h>
#include <string>
#include <vector>
#include <unordered_map>

extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"

#include "postgres.h"
#include "fmgr.h"
#include "access/htup_details.h" // 提供 heap_getattr
#include "utils/lsyscache.h"     // 提供 getTypeOutputInfo
#include "utils/syscache.h"
}

#define MAX_CACHE_SIZE 100000

#include <chrono>
#include <unordered_map>

class MemoryManager {
public:
    MemoryManager() {}
    ~MemoryManager() {}
    static Args* LoadOneRow(const std::string& table_name, size_t row_index);
    static void SPILoadOneRow(HeapTuple& tuple, TupleDesc& tupdesc, const std::string& table_name, size_t row_index);
    static Args* Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim);

    // to be done
    int current_func_call{-1};
    int is_last_call{2};
    float** ins_cache_data;
    MVec** ins_cache;
    int out_cache_size{0};
    float* out_cache_data;
    int output_index{0};
    Args* ins_buffer{nullptr};

    std::vector<std::string> sample_path;
    
    // Execution time tracking
    std::unordered_map<std::string, long long> execution_time_map;  // Agent name to accumulated execution time (in microseconds)
    std::unordered_map<std::string, int> execution_count_map;       // Agent name to execution count
    
    // Utility function to print timing statistics
    void PrintTimingStats();
};

// agent result: the result of the agent execution
enum class AgentAction {
    START,
    SUCCESS,
    FAILURE,
    PERCEPTION,
    ORCHESTRATION,
    OPTIMIZATION,
    EXECUTION,
    EVALUATION,
    SCHEDULE
};

enum class TaskType {
    IMAGE_CLASSIFICATION,
    SERIES,
    NLP
    // to be done
};

typedef struct Task {
    TaskType task_type;
    char* model;
    char* cuda;
    char* table_name;

    // =========================


    int64_t input_start_index;
    int64_t input_end_index;
    int64_t output_start_index;
    int64_t output_end_index;
} Task;

// TODO: 可以考虑根据不同任务类型进行继承类划分
typedef struct TaskInfo {
    TaskType task_type;
    char* table_name;
    char* model_name;
    char* cuda_name;
} TaskInfo;

typedef struct VecAggState {
    MemoryContext ctx;
    List* ins;
    List* outs;
    int batch_i;
    int prcsd_batch_n;
    char* model;
    char* cuda;
    int nxt_csr;
    int64_t pre_time;   // ms
    int64_t infer_time; // ms
    int64_t post_time;  // ms
} VecAggState;

// agent state: present the current state of the agent
typedef struct AgentState {
    FunctionCallInfo fcinfo;
    VecAggState current_state;
    AgentAction last_action;
    int current_start_index;
    int current_end_index;
    int current_task_id;    // task_id即为task_list的下标
    List* task_list;
    std::vector<TaskInfo> task_info;
    
    // to be done
}AgentState;
// base agent node
class BaseAgentNode {
public:
    virtual ~BaseAgentNode() = default;

    // 核心接口：输入当前状态，修改状态
    virtual AgentAction Execute(std::shared_ptr<AgentState> state) = 0;
    
    // 获取节点名称
    virtual std::string Name() const = 0;
    
    // Timed execution wrapper
    AgentAction ExecuteWithTiming(std::shared_ptr<AgentState> state);
};

// perception agent: NL =====> embedding vector
class PerceptionAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    void LoadMVecData(MVec* input);

    std::string Name() const override {
        return "PerceptionAgent";
    }
};

// orchestration agent: model selection, resource management
class OrchestrationAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    void SPIRegisterProcess();
    void TaskInit(std::shared_ptr<AgentState> state);
    std::string SelectModel(std::shared_ptr<AgentState> state, TaskType task_type);
    bool InitModel(const char* model_name, bool from_select_model);

    std::string Name() const override {
        return "OrchestrationAgent";
    }
};

// optimization agent: planning tree optimization
class OptimizationAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    std::string Name() const override {
        return "OptimizationAgent";
    }
};

// execution agent: execute the plan
class ExecutionAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    std::string Name() const override {
        return "ExecutionAgent";
    }
};

// evaluation agent: evaluate time and return the result
class EvaluationAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    std::string Name() const override {
        return "EvaluationAgent";
    }
};

// schedule agent: manage all the agents
class ScheduleAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    std::string Name() const override {
        return "ScheduleAgent";
    }
};

// Structure to hold the result of model analysis
struct ModelAnalysisResult {
    long long mac_count;      // Multiply-Accumulate operations
    long long param_count;    // Total number of parameters
    size_t param_size_bytes;  // Total size of parameters in bytes
};

// Function to analyze model characteristics: MAC, parameter count, and parameter size
ModelAnalysisResult AnalyzeModelWithInference(const std::string& model_name);

#endif