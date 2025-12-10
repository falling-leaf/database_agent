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

Datum
api_agent(PG_FUNCTION_ARGS) {
    PerceptionAgent      perception_agent_;     // perception: 将大任务拆分为多个TaskType
    OrchestrationAgent   orchestration_agent_;  // orchestration: 配置元信息，将TaskInfo转化为Task
    OptimizationAgent    optimization_agent_;
    ExecutionAgent       execution_agent_;
    EvaluationAgent      evaluation_agent_;
    ScheduleAgent        schedule_agent_;
    auto state_ = std::make_shared<AgentState>();
    state_->fcinfo = fcinfo;
    std::map<AgentAction, std::function<AgentAction(std::shared_ptr<AgentState>)>> func_map_;
    func_map_[AgentAction::PERCEPTION] = [&perception_agent_](std::shared_ptr<AgentState> s) {return perception_agent_.Execute(s);};
    func_map_[AgentAction::ORCHESTRATION] = [&orchestration_agent_](std::shared_ptr<AgentState> s) {return orchestration_agent_.Execute(s);};
    func_map_[AgentAction::OPTIMIZATION] = [&optimization_agent_](std::shared_ptr<AgentState> s) {return optimization_agent_.Execute(s);};
    func_map_[AgentAction::EXECUTION] = [&execution_agent_](std::shared_ptr<AgentState> s) {return execution_agent_.Execute(s);};
    func_map_[AgentAction::EVALUATION] = [&evaluation_agent_](std::shared_ptr<AgentState> s) {return evaluation_agent_.Execute(s);};
    func_map_[AgentAction::SCHEDULE] = [&schedule_agent_](std::shared_ptr<AgentState> s) {return schedule_agent_.Execute(s);};

    AgentAction next_action = initialize_state(state_);
    while (next_action != AgentAction::SUCCESS && next_action != AgentAction::FAILURE && next_action != AgentAction::START) {
        if (func_map_.find(next_action) != func_map_.end()) {
            next_action = func_map_[next_action](state_);
        } else {
            ereport(ERROR, (errmsg("api_agent: Unknown action %d", static_cast<int>(next_action))));
            PG_RETURN_BOOL(false);
        }
    }
    if (next_action == AgentAction::SUCCESS) {
        PG_RETURN_BOOL(true);
    } else {
    PG_RETURN_BOOL(false);
    }
}

#ifdef __cplusplus
}
#endif
