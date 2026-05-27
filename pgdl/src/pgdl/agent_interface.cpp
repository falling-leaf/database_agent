/**
 * @file agent_interface.cpp
 * @brief PostgreSQL entry points and global state for the PGDL agent system
 *
 * Structure:
 *   1. Includes (agent_interface.h + auxiliary headers)
 *   2. extern "C" block for PostgreSQL headers + PG_FUNCTION_INFO_V1 declarations
 *   3. Global state definitions (C++ types, must be OUTSIDE extern "C")
 *   4. Internal utility functions (C++ types)
 *   5. extern "C" block for PostgreSQL function implementations
 */

#include "agent_interface.h"

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

/* ====================================================================== */
/*  PostgreSQL headers & function declarations (C linkage)                */
/* ====================================================================== */
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
#ifdef __cplusplus
}
#endif

/* ====================================================================== */
/*  Global state (C++ types — must stay OUTSIDE extern "C")               */
/* ====================================================================== */
MemoryManager memory_manager;
std::shared_ptr<AgentState> state_ = std::make_shared<AgentState>();

PerceptionAgent      perception_agent_;     // perception: split task into TaskTypes
OrchestrationAgent   orchestration_agent_;  // orchestration: configure metadata, TaskInfo -> Task
OptimizationAgent    optimization_agent_;
ExecutionAgent       execution_agent_;
EvaluationAgent      evaluation_agent_;
ScheduleAgent        schedule_agent_;
std::map<AgentAction, std::function<AgentAction(std::shared_ptr<AgentState>)>> func_map_;

/* ====================================================================== */
/*  Internal utility functions                                            */
/* ====================================================================== */
void initialize_state(std::shared_ptr<AgentState> state) {
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
}

void reset_global_memory_state() {
    // 1. Deep clean: free each element in arrays
    if (memory_manager.ins_cache != NULL) {
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

    if (memory_manager.ins_cache_data != NULL) {
        pfree(memory_manager.ins_cache_data);
        memory_manager.ins_cache_data = NULL;
    }

    // 3. Clean ins2_cache
    if (memory_manager.ins2_cache != NULL) {
        if (memory_manager.current_func_call > 0) {
            for (int i = 0; i <= memory_manager.current_func_call; i++) {
                if (memory_manager.ins2_cache[i] != NULL) {
                    pfree(memory_manager.ins2_cache[i]);
                    memory_manager.ins2_cache[i] = NULL;
                }
                memory_manager.ins2_cache_data[i] = NULL;
            }
        }
        pfree(memory_manager.ins2_cache);
        memory_manager.ins2_cache = NULL;
    }

    if (memory_manager.out_cache_data != NULL) {
        pfree(memory_manager.out_cache_data);
        memory_manager.out_cache_data = NULL;
    }

    // 4. Clean out_cache_string_data
    if (memory_manager.out_cache_string_data != NULL) {
        if (memory_manager.current_func_call > 0) {
            for (int i = 0; i <= memory_manager.current_func_call; i++) {
                if (memory_manager.out_cache_string_data[i] != NULL) {
                    pfree(memory_manager.out_cache_string_data[i]);
                    memory_manager.out_cache_string_data[i] = NULL;
                }
            }
        }
        pfree(memory_manager.out_cache_string_data);
        memory_manager.out_cache_string_data = NULL;
    }

    // Clean global state_ persistent containers to prevent stale pointers on reuse
    if (state_) {
        state_->task_info.clear();
        state_->task_list = NIL;
        state_->current_task_id = 0;
        state_->current_state.ins = NIL;
        state_->current_state.outs = NIL;
        state_->current_start_index = 0;
        state_->current_end_index = 0;
    }

    // Reset counters
    memory_manager.current_func_call = -1;
    memory_manager.is_last_call = 2;

    // Clean ins_buffer
    if (memory_manager.ins_buffer != NULL) {
        pfree(memory_manager.ins_buffer);
        memory_manager.ins_buffer = NULL;
    }

    memory_manager.sample_path.clear();
    memory_manager.sample_path.shrink_to_fit();
    memory_manager.execution_time_map.clear();
    memory_manager.execution_count_map.clear();

    elog(INFO, "Global memory state deeply cleaned.");
}

/* ====================================================================== */
/*  PostgreSQL entry points (C linkage)                                   */
/* ====================================================================== */
#ifdef __cplusplus
extern "C" {
#endif

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
    Datum* elems;
    ArrayType  *result_array;
    elems = (Datum*)palloc(sizeof(Datum) * memory_manager.out_cache_size);
    for(int i = 0; i < memory_manager.out_cache_size; i++) {
        elems[i] = Float8GetDatum((double)memory_manager.out_cache_data[i]);
    }

    result_array = construct_array(elems, memory_manager.out_cache_size, FLOAT8OID, 8, true, 'd');
    reset_global_memory_state();
    PG_RETURN_FLOAT8(memory_manager.out_cache_data[0]);
}

Datum db_agent_sfinal(PG_FUNCTION_ARGS) {
    memory_manager.is_last_call = 1;
    state_->last_action = AgentAction::PERCEPTION;
    AgentAction next_action = AgentAction::SCHEDULE;

    if (true) {
        while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
            if (func_map_.find(next_action) != func_map_.end()) {
                next_action = func_map_[next_action](state_);
            } else {
                ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
            }
        }
    }

    bool need_reasoning = true;
    bool need_musique = true;
    for (auto& task_unit : state_->task_info) {
        if (task_unit.task_type != TaskType::REASONING) {
            need_reasoning = false;
        }
        if (task_unit.task_type != TaskType::MUSIQUE) {
            need_musique = false;
        }
    }
    if (need_reasoning) {
        temp_addition_function(state_);
    } else if (need_musique) {
        temp_addition_function_musique(state_);
    }

    // Build result array (output_type 0 and 1 share the same logic)
    Datum* elems = (Datum*)palloc(sizeof(Datum) * memory_manager.out_cache_size);
    for(int i = 0; i < memory_manager.out_cache_size; i++) {
        elems[i] = Float8GetDatum((double)memory_manager.out_cache_data[i]);
    }
    ArrayType *result_array = construct_array(elems, memory_manager.out_cache_size, FLOAT8OID, 8, true, 'd');
    memory_manager.PrintTimingStats();
    reset_global_memory_state();
    PG_RETURN_ARRAYTYPE_P(result_array);
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
    if (memory_manager.is_last_call == 0 && memory_manager.output_index >= memory_manager.out_cache_size) {
        memory_manager.PrintTimingStats();
        reset_global_memory_state();
    }
    PG_RETURN_FLOAT8(ret);
}

#ifdef __cplusplus
}
#endif
