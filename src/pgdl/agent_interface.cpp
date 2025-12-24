#include "model_manager.h"
#include "model_selection.h"
#include "model_agent.h"

#include "unistd.h"
#include "myfunc.h"
#include "embedding.h"
#include "model_utils.h"
#include "vector.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "httprequest.hpp"
#include <memory>
#include <map>
#include <functional>


#ifdef __cplusplus
extern "C" {
#endif
#include "utils/builtins.h"
#include "utils/formatting.h"
#include "catalog/pg_type_d.h"
#include "catalog/namespace.h"
#include "common/fe_memutils.h"
#include "access/tableam.h"
#include "libpq/pqformat.h"
#include "utils/syscache.h"

PG_FUNCTION_INFO_V1(db_agent_final);
PG_FUNCTION_INFO_V1(db_agent_sfunc);
MemoryManager memory_manager;
auto state_ = std::make_shared<AgentState>();

PerceptionAgent      perception_agent_;     // perception: 将大任务拆分为多个TaskType
OrchestrationAgent   orchestration_agent_;  // orchestration: 配置元信息，将TaskInfo转化为Task
OptimizationAgent    optimization_agent_;
ExecutionAgent       execution_agent_;
EvaluationAgent      evaluation_agent_;
ScheduleAgent        schedule_agent_;
std::map<AgentAction, std::function<AgentAction(std::shared_ptr<AgentState>)>> func_map_;

void initialize_state(std::shared_ptr<AgentState> state) {
    // Initialize the agent state
    if (memory_manager.out_cache_data != NULL) {
        pfree(memory_manager.out_cache_data);
        memory_manager.out_cache_data = NULL;
    }
    memory_manager.out_cache_size = 0;
    func_map_[AgentAction::PERCEPTION] = [](std::shared_ptr<AgentState> s) {return perception_agent_.Execute(s);};
    func_map_[AgentAction::ORCHESTRATION] = [](std::shared_ptr<AgentState> s) {return orchestration_agent_.Execute(s);};
    func_map_[AgentAction::OPTIMIZATION] = [](std::shared_ptr<AgentState> s) {return optimization_agent_.Execute(s);};
    func_map_[AgentAction::EXECUTION] = [](std::shared_ptr<AgentState> s) {return execution_agent_.Execute(s);};
    func_map_[AgentAction::EVALUATION] = [](std::shared_ptr<AgentState> s) {return evaluation_agent_.Execute(s);};
    func_map_[AgentAction::SCHEDULE] = [](std::shared_ptr<AgentState> s) {return schedule_agent_.Execute(s);};
    return;
}

void reset_global_memory_state() {
    // 1. 深度清理：释放数组内的每个元素
    if (memory_manager.ins_cache != NULL) {
        // 注意：ins_cache_data 指向的是 ins_cache 内存块内部，不需要单独 pfree 数据部分
        // 只需要 free ins_cache[i] 即可
        if (memory_manager.current_func_call > 0) {
            for (int i = 0; i <= memory_manager.current_func_call; i++) {
                if (memory_manager.ins_cache[i] != NULL) {
                    pfree(memory_manager.ins_cache[i]);
                    memory_manager.ins_cache[i] = NULL;
                }
                memory_manager.ins_cache_data[i] = NULL;
            }
        }
        pfree(memory_manager.ins_cache);
        memory_manager.ins_cache = NULL;
    }

    // 释放数据指针数组本身
    if (memory_manager.ins_cache_data != NULL) {
        pfree(memory_manager.ins_cache_data);
        memory_manager.ins_cache_data = NULL;
    }

    // 【关键修正 1】: 清理全局 state_ 中的持久化容器
    // 否则第二次连接复用时，task_info 会残留上一次的野指针
    if (state_) {
        state_->task_info.clear(); 
        state_->task_list = NIL;  // 清空 PG 的 List 引用
        state_->current_task_id = 0;
        
        // 同样清理执行状态中的列表，防止悬垂指针
        state_->current_state.ins = NIL;
        state_->current_state.outs = NIL;
        state_->current_start_index = 0;
        state_->current_end_index = 0;
    }

    // 2. 重置计数器
    memory_manager.current_func_call = -1;
    memory_manager.is_last_call = false;

    memory_manager.sample_path.clear();          // size变为0，但capacity仍为1000
    memory_manager.sample_path.shrink_to_fit();
    
    elog(INFO, "Global memory state deeply cleaned.");
}

Datum
db_agent_sfunc(PG_FUNCTION_ARGS) {
    state_->fcinfo = fcinfo;

    if (memory_manager.current_func_call == -1)
        initialize_state(state_);

    state_->last_action = AgentAction::START;
    AgentAction next_action = AgentAction::SCHEDULE;

    while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
        if (func_map_.find(next_action) != func_map_.end()) {
            next_action = func_map_[next_action](state_);
        } else {
            ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
            // 发生错误时，必须清理内存，否则下次查询会崩溃
            reset_global_memory_state();
            PG_RETURN_BOOL(false);
        }
    }

    if (next_action == AgentAction::SUCCESS) {
        PG_RETURN_BOOL(true);
    } else {
        PG_RETURN_BOOL(false);
    }
}

Datum db_agent_final(PG_FUNCTION_ARGS) {
    memory_manager.is_last_call = true;
    state_->last_action = AgentAction::PERCEPTION;
    AgentAction next_action = AgentAction::SCHEDULE;

    while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
        if (func_map_.find(next_action) != func_map_.end()) {
            next_action = func_map_[next_action](state_);
        } else {
            ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
        }
    }

    // Process the out_cache_data - read from memory manager
    // The actual results are stored in memory_manager.out_cache_data array
    // with size memory_manager.out_cache_size
    // For now, we just return true as the aggregate expects boolean
    // A separate function could be created to retrieve the multiple results
    Datum* elems;       // 存放数据的 Datum 数组
    ArrayType  *result_array;   // PostgreSQL 的数组对象
    
    // 步骤 A: Prepare the data array based on actual cache size
    // Allocate memory for the datum array based on the actual number of results
    elems = (Datum*)palloc(sizeof(Datum) * memory_manager.out_cache_size);
    
    // Fill the datum array with values from the out_cache_data
    for(int i = 0; i < memory_manager.out_cache_size; i++) {
        elems[i] = Float8GetDatum((double)memory_manager.out_cache_data[i]);
    }

    // 步骤 B: 构造数组
    // 参数说明:
    // 1. elems: 数据指针
    // 2. memory_manager.out_cache_size: 元素个数
    // 3. FLOAT8OID: 元素类型ID (float8)
    // 4. 8: 元素长度 (sizeof(double))
    // 5. true: float8 是 pass-by-value (传值)
    // 6. 'd': double 的对齐方式
    result_array = construct_array(elems, memory_manager.out_cache_size, FLOAT8OID, 8, true, 'd');

    // 6. 返回数组
   
    reset_global_memory_state();
    PG_RETURN_ARRAYTYPE_P(result_array);
}

#ifdef __cplusplus
}
#endif
