#pragma once
#ifndef _AGENT_INTERFACE_H_
#define _AGENT_INTERFACE_H_

/**
 * @file agent_interface.h
 * @brief Interface declarations shared between agent_interface.cpp and model_agent.cpp
 * 
 * Provides clean extern declarations for global state, agent instances,
 * and internal utility functions. This separates the PostgreSQL entry
 * point layer from the agent implementation layer.
 */

#include "model_agent.h"

// =============================================================================
// Global state
// =============================================================================
extern MemoryManager memory_manager;
extern std::shared_ptr<AgentState> state_;

// Agent singletons
extern PerceptionAgent      perception_agent_;
extern OrchestrationAgent   orchestration_agent_;
extern OptimizationAgent    optimization_agent_;
extern ExecutionAgent       execution_agent_;
extern EvaluationAgent      evaluation_agent_;
extern ScheduleAgent        schedule_agent_;

// Action-to-function dispatch map
extern std::map<AgentAction, std::function<AgentAction(std::shared_ptr<AgentState>)>> func_map_;

// =============================================================================
// Internal utility functions
// =============================================================================
void initialize_state(std::shared_ptr<AgentState> state);
void reset_global_memory_state();

// Reasoning pipeline helper (called from db_agent_sfinal)
void temp_addition_function(std::shared_ptr<AgentState> state);
void temp_addition_function_musique(std::shared_ptr<AgentState> state);

// External tools called by the reasoning pipeline
std::string CallToolReaderDecoder(std::string id_str);

#endif /* _AGENT_INTERFACE_H_ */
