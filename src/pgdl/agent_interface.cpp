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

PG_FUNCTION_INFO_V1(api_agent);
MemoryManager memory_manager;
auto state_ = std::make_shared<AgentState>();

PerceptionAgent      perception_agent_;     // perception: 将大任务拆分为多个TaskType
OrchestrationAgent   orchestration_agent_;  // orchestration: 配置元信息，将TaskInfo转化为Task
OptimizationAgent    optimization_agent_;
ExecutionAgent       execution_agent_;
EvaluationAgent      evaluation_agent_;
ScheduleAgent        schedule_agent_;

void reset_global_memory_state() {
    // 1. 深度清理：释放数组内的每个元素
    if (memory_manager.ins_cache != NULL) {
        // 注意：ins_cache_data 指向的是 ins_cache 内存块内部，不需要单独 pfree 数据部分
        // 只需要 free ins_cache[i] 即可
        if (memory_manager.total_func_call > 0) {
            for (int i = 0; i < memory_manager.total_func_call; i++) {
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
    memory_manager.total_func_call = 0;
    memory_manager.current_func_call = 0;
    
    elog(INFO, "Global memory state deeply cleaned.");
}

Datum
api_agent(PG_FUNCTION_ARGS) {
    state_->fcinfo = fcinfo;
    
    // ... func_map initialization ...
    std::map<AgentAction, std::function<AgentAction(std::shared_ptr<AgentState>)>> func_map_;
    func_map_[AgentAction::PERCEPTION] = [](std::shared_ptr<AgentState> s) {return perception_agent_.Execute(s);};
    func_map_[AgentAction::ORCHESTRATION] = [](std::shared_ptr<AgentState> s) {return orchestration_agent_.Execute(s);};
    func_map_[AgentAction::OPTIMIZATION] = [](std::shared_ptr<AgentState> s) {return optimization_agent_.Execute(s);};
    func_map_[AgentAction::EXECUTION] = [](std::shared_ptr<AgentState> s) {return execution_agent_.Execute(s);};
    func_map_[AgentAction::EVALUATION] = [](std::shared_ptr<AgentState> s) {return evaluation_agent_.Execute(s);};
    func_map_[AgentAction::SCHEDULE] = [](std::shared_ptr<AgentState> s) {return schedule_agent_.Execute(s);};

    // 【防御性编程】: 如果 total 不为0但 current 也是0，说明上次可能异常退出了，强制清理
    // 这种情况比较少见，但在开发调试阶段很有用
    if (memory_manager.total_func_call != 0 && memory_manager.current_func_call == 0) {
         // 注意：这里需要谨慎，因为正常的第一行逻辑 current_func_call 在 PerceptionAgent 内部才变
         // 所以主要依靠下面的结束清理逻辑。
    }

    AgentAction next_action = initialize_state(state_);
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
        // 检查是否是本批次的最后一行
        // PerceptionAgent 中逻辑：row 1 -> current=0; row 50 -> current=49. Total=50.
        // 所以当 current + 1 == total 时，说明这是最后一行。
        if (memory_manager.total_func_call > 0 && 
            memory_manager.current_func_call + 1 >= memory_manager.total_func_call) {
            
            // 【关键修复】: 此时显式释放全局内存，防止留给下一次查询
            reset_global_memory_state();
        }
        PG_RETURN_BOOL(true);
    } else {
        // 任务失败 (FAILURE)，也要清理环境以便重试
        reset_global_memory_state();
        PG_RETURN_BOOL(false);
    }
}

#ifdef __cplusplus
}
#endif
