#pragma once
#ifndef _MODEL_AGENT_H_
#define _MODEL_AGENT_H_

#include "env.h"
#include "model_manager.h"
#include "batch_interface.h"
#include "myfunc.h"

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

class MemoryManager {
public:
    MemoryManager() {}
    ~MemoryManager() {}
    static Args* LoadOneRow(const std::string& table_name, size_t row_index);
    static void SPILoadOneRow(HeapTuple& tuple, TupleDesc& tupdesc, const std::string& table_name, size_t row_index);
    static Args* Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim);

    // to be done
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

typedef struct Task {
    char* model;
    char* cuda;
    char* table_name;
    int64_t input_start_index;
    int64_t input_length;
} Task;

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
    VecAggState current_state;
    AgentAction last_action;
    std::vector<Task> task_list;
    
    // to be done
}AgentState;

AgentAction initialize_state(std::shared_ptr<AgentState> state);

// base agent node
class BaseAgentNode {
public:
    virtual ~BaseAgentNode() = default;

    // 核心接口：输入当前状态，修改状态
    virtual AgentAction Execute(std::shared_ptr<AgentState> state) = 0;
    
    // 获取节点名称
    virtual std::string Name() const = 0;
};

// perception agent: NL =====> embedding vector
class PerceptionAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    std::string Name() const override {
        return "PerceptionAgent";
    }
};

// orchestration agent: model selection, resource management
class OrchestrationAgent : public BaseAgentNode {
public:
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    void SPIRegisterProcess();

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

#endif