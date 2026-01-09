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
PG_FUNCTION_INFO_V1(db_agent_sfinal);
PG_FUNCTION_INFO_V1(db_agent_sfunc);
PG_FUNCTION_INFO_V1(db_agent_mfinalfunc);
PG_FUNCTION_INFO_V1(db_agent_msfunc);
PG_FUNCTION_INFO_V1(db_agent_minvfunc);

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
    memory_manager.output_index = 0;
    func_map_[AgentAction::PERCEPTION] = [](std::shared_ptr<AgentState> s) {return perception_agent_.ExecuteWithTiming(s);};
    func_map_[AgentAction::ORCHESTRATION] = [](std::shared_ptr<AgentState> s) {return orchestration_agent_.ExecuteWithTiming(s);};
    func_map_[AgentAction::OPTIMIZATION] = [](std::shared_ptr<AgentState> s) {return optimization_agent_.ExecuteWithTiming(s);};
    func_map_[AgentAction::EXECUTION] = [](std::shared_ptr<AgentState> s) {return execution_agent_.ExecuteWithTiming(s);};
    func_map_[AgentAction::EVALUATION] = [](std::shared_ptr<AgentState> s) {return evaluation_agent_.ExecuteWithTiming(s);};
    func_map_[AgentAction::SCHEDULE] = [](std::shared_ptr<AgentState> s) {return schedule_agent_.ExecuteWithTiming(s);};
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
    memory_manager.is_last_call = 2;

    // 【关键修正 2】: 清理 ins_buffer
    if (memory_manager.ins_buffer != NULL) {
        pfree(memory_manager.ins_buffer);
        memory_manager.ins_buffer = NULL;
    }

    memory_manager.sample_path.clear();          // size变为0，但capacity仍为1000
    memory_manager.sample_path.shrink_to_fit();
    
    // Clear timing statistics
    memory_manager.execution_time_map.clear();
    memory_manager.execution_count_map.clear();
    
    elog(INFO, "Global memory state deeply cleaned.");
}

Datum
db_agent_sfunc(PG_FUNCTION_ARGS) {
    if (memory_manager.is_last_call == 3) {
        elog(INFO, "last call crashed, reset");
        memory_manager.PrintTimingStats();
        reset_global_memory_state();
    }
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
    state_->last_action = AgentAction::PERCEPTION;
    AgentAction next_action = AgentAction::SCHEDULE;

    if (memory_manager.current_func_call % 32 != 31) {
        while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
            if (func_map_.find(next_action) != func_map_.end()) {
                next_action = func_map_[next_action](state_);
            } else {
                ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
            }
        }
    }
    Datum* elems;       // 存放数据的 Datum 数组
    ArrayType  *result_array;   // PostgreSQL 的数组对象
    elems = (Datum*)palloc(sizeof(Datum) * memory_manager.out_cache_size);
    for(int i = 0; i < memory_manager.out_cache_size; i++) {
        elems[i] = Float8GetDatum((double)memory_manager.out_cache_data[i]);
    }

    result_array = construct_array(elems, memory_manager.out_cache_size, FLOAT8OID, 8, true, 'd');
    reset_global_memory_state();
    // PG_RETURN_ARRAYTYPE_P(result_array);
    PG_RETURN_FLOAT8(memory_manager.out_cache_data[0]);
}

Datum db_agent_sfinal(PG_FUNCTION_ARGS) {
    memory_manager.is_last_call = 1;
    state_->last_action = AgentAction::PERCEPTION;
    AgentAction next_action = AgentAction::SCHEDULE;

    if (memory_manager.current_func_call % 32 != 31) {
        while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
            if (func_map_.find(next_action) != func_map_.end()) {
                next_action = func_map_[next_action](state_);
            } else {
                ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
            }
        }
    }
    Datum* elems;       // 存放数据的 Datum 数组
    ArrayType  *result_array;   // PostgreSQL 的数组对象
    elems = (Datum*)palloc(sizeof(Datum) * memory_manager.out_cache_size);
    for(int i = 0; i < memory_manager.out_cache_size; i++) {
        elems[i] = Float8GetDatum((double)memory_manager.out_cache_data[i]);
    }

    result_array = construct_array(elems, memory_manager.out_cache_size, FLOAT8OID, 8, true, 'd');
    memory_manager.PrintTimingStats();
    reset_global_memory_state();
    PG_RETURN_ARRAYTYPE_P(result_array);
    // PG_RETURN_FLOAT8(memory_manager.out_cache_data[0]);
}

Datum db_agent_msfunc(PG_FUNCTION_ARGS) {
    elog(INFO, "db_agent_msfunc, is_last_call = %d, output_index = %d, memory_manager.current_func_call = %d", memory_manager.is_last_call, memory_manager.output_index, memory_manager.current_func_call);
    if (memory_manager.is_last_call == 3) {
        elog(INFO, "last call crashed, reset");
        memory_manager.PrintTimingStats();
        reset_global_memory_state();
    }
    state_->fcinfo = fcinfo;
    memory_manager.is_last_call = 2;

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

Datum db_agent_minvfunc(PG_FUNCTION_ARGS) {
    elog(INFO, "db_agent_minvfunc, is_last_call = %d, output_index = %d, memory_manager.current_func_call = %d", memory_manager.is_last_call, memory_manager.output_index, memory_manager.current_func_call);
    if (memory_manager.is_last_call == 3) {
        memory_manager.is_last_call = 1;
    }
    PG_RETURN_BOOL(true);
}

Datum db_agent_mfinalfunc(PG_FUNCTION_ARGS) {
    elog(INFO, "db_agent_mfinalfunc, is_last_call = %d, output_index = %d, memory_manager.current_func_call = %d", memory_manager.is_last_call, memory_manager.output_index, memory_manager.current_func_call);
    if (memory_manager.is_last_call == 1) {
        state_->last_action = AgentAction::PERCEPTION;
        AgentAction next_action = AgentAction::SCHEDULE;  
        if (memory_manager.current_func_call % 32 != 31) {
            while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
                if (func_map_.find(next_action) != func_map_.end()) {
                    next_action = func_map_[next_action](state_);
                } else {
                    ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
                }
            }
        }
        memory_manager.is_last_call = 0;
    } else {
        memory_manager.is_last_call = 3;
    }
    double ret = (double)memory_manager.out_cache_data[memory_manager.output_index++];
    // 原理在于：out_cache_size更新速度一定跑的比output_index快
    // 问题：当整条sql后面有limit时，执行逻辑与预想不同；该方法只能支持有界完整的数据库查询指令
    if (memory_manager.is_last_call == 0 && memory_manager.output_index >= memory_manager.out_cache_size) {
        // Print timing statistics
        memory_manager.PrintTimingStats();
        reset_global_memory_state();
    }
    PG_RETURN_FLOAT8(ret);
}

#ifdef __cplusplus
}
#endif
