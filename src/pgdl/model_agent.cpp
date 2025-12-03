#include "model_agent.h"
#include "model_manager.h"
#include "model_utils.h"
#include "spi_connection.h"
#include "md5.h"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>


extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "miscadmin.h"

extern void infer_batch_internal(VecAggState *state, bool ret_float8);
}
extern void register_callback();

Args* MemoryManager::Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim)
{
    int mvec_oid = 0;
    // get_mvec_oid定义于model_uitls.cpp, 用于获取mvec类型的oid
    if (!get_mvec_oid(mvec_oid))
    {
        ereport(ERROR, (errmsg("get mvec oid error!")));
        return nullptr;
    }
    // slice中仅处理一列数据（mvec）
    if (tupdesc->natts != dim)
    {
        ereport(ERROR, (errmsg("tuple dimension %d does not match expected dimension %d!",
                               tupdesc->natts, dim)));
        return nullptr;
    }
    Args* vec = (Args*) palloc0(sizeof(Args) * dim);  // 仍然用 PG 的 palloc

    // =============================================================================

    for (int i = start; i < dim + start; i++) {
        // 1. 获取列的类型 OID
        // tupdesc->attrs 是 0-based 数组，所以直接用 i (假设 start=0 时 i=0 对应第1列)
        Oid argtype = tupdesc->attrs[i].atttypid;

        // 2. 从元组中获取二进制数据 (Datum)
        // heap_getattr 的 attnum 是 1-based，所以用 i + 1
        bool isnull;
        Datum datum = heap_getattr(tuple, i + 1, tupdesc, &isnull);

        // 3. 处理 NULL 值（防止读取垃圾数据）
        if (isnull) {
            vec[i - start].ptr = nullptr;
            vec[i - start].floating = 0.0;
            vec[i - start].integer = 0;
            continue;
        }

        // 4. 根据类型转换 Datum
        if (argtype == INT4OID) {
            vec[i - start].integer = DatumGetInt32(datum);
        }
        else if (argtype == INT2OID) {
            vec[i - start].integer = DatumGetInt16(datum);
        }
        else if (argtype == INT8OID) {
            vec[i - start].integer = DatumGetInt64(datum);
        }
        else if (argtype == FLOAT4OID) {
            vec[i - start].floating = (double)DatumGetFloat4(datum);
        }
        else if (argtype == FLOAT8OID) {
            vec[i - start].floating = DatumGetFloat8(datum);
        }
        else if (argtype == TEXTOID || argtype == VARCHAROID) {
            // TextDatumGetCString 会分配新内存(palloc)，返回 C 风格字符串
            char* cur_text = TextDatumGetCString(datum);
            vec[i - start].ptr = cur_text;
        }
        else if (argtype == CSTRINGOID) {
            // DatumGetCString 返回的指针可能直接指向 Tuple 内部，
            // 建议使用 pstrdup 复制一份，保证生命周期安全
            char* cur_cstring = pstrdup(DatumGetCString(datum));
            vec[i - start].ptr = cur_cstring;
        }
        else if (argtype == NUMERICOID) {
            // Numeric 转换逻辑保持不变，利用 PG 内部函数
            float8 num_float = DatumGetFloat8(DirectFunctionCall1(numeric_float8, datum));
            vec[i - start].floating = num_float;
        }
        else if (argtype == mvec_oid) {
            // MVec 通常是引用类型，使用 DatumGetPointer
            // 如果你有自定义的 DatumGetMVec 宏也可以使用，本质通常是一样的
            // elog(INFO, "found mvec column at %d", i + 1);
            MVec* cur_mvec = (MVec*)DatumGetPointer(datum);
            vec[i - start].ptr = cur_mvec;
        }
        else {
            ereport(ERROR, (errmsg("type %u not supported at column %d!", argtype, i + 1)));
        }
    }
    return vec;
}

void MemoryManager::SPILoadOneRow(HeapTuple& tuple, TupleDesc& tupdesc, const std::string& table_name, size_t row_index) {
    // 保存当前的内存上下文（这是调用者的上下文，比如 ExecutorState）
    MemoryContext caller_context = CurrentMemoryContext;

    SPIConnector spi_connector; // 连接 SPI
    // 注意：SPI_connect 会将 CurrentMemoryContext 切换为 SPI 的上下文

    std::string sql_str = "SELECT * FROM " + table_name + " LIMIT 1 OFFSET $1";
    SPISqlWrapper sql(spi_connector, sql_str, 1);

    int64 offset = static_cast<int64>(row_index);

    if (!sql.Bind(1, INT8OID, Int64GetDatum(offset)) || !sql.Execute())
    {
        elog(WARNING, "LoadOneRow: failed...");
        tuple = nullptr;
        tupdesc = nullptr;
        return;
    }

    if (SPI_processed != 1)
    {
        elog(INFO, "LoadOneRow: no row found...");
        tuple = nullptr;
        tupdesc = nullptr;
        return;
    }

    // ================== 关键修改 ==================
    
    // 1. 获取 SPI 内部的指针
    HeapTuple spi_tuple = SPI_tuptable->vals[0];
    TupleDesc spi_tupdesc = SPI_tuptable->tupdesc;

    // 2. 临时切回调用者的内存上下文，以便分配的内存能在 SPI_finish 后存活
    MemoryContext old_ctx = MemoryContextSwitchTo(caller_context);

    // 3. 执行深拷贝
    // heap_copytuple 会分配新内存并复制数据
    tuple = heap_copytuple(spi_tuple);
    // CreateTupleDescCopy 会分配新内存并复制描述符
    tupdesc = CreateTupleDescCopy(spi_tupdesc);

    // 4. 切回 SPI 上下文（以便 spi_connector 析构时能正常清理环境，虽然大部分时候 SPI_finish 会重置它）
    MemoryContextSwitchTo(old_ctx);

    // =============================================
    
    return; // spi_connector 析构，释放 SPI 内部内存，但我们拷贝的数据在 caller_context 中，是安全的
}

Args* MemoryManager::LoadOneRow(const std::string& table_name, size_t row_index)
{

    HeapTuple   tuple;
    TupleDesc   tupdesc;
    SPILoadOneRow(tuple, tupdesc, table_name, row_index);
    int         ncols   = tupdesc->natts;
    // elog(INFO, "start converting row to vector, ncols=%d", ncols);

    Args* vec = MemoryManager::Tuple2Vec(tuple, tupdesc, 0, ncols);

    // elog(INFO, "finished converting row to vector");
    return vec;
}


AgentAction initialize_state(std::shared_ptr<AgentState> state) {
    // Initialize the agent state
    state->last_action = AgentAction::START;
    // to be done
    return AgentAction::SCHEDULE;
}

// perception agent: NL =====> embedding vector
AgentAction PerceptionAgent::Execute(std::shared_ptr<AgentState> state) {
    List* inputs = get_inputs_();
    TaskInfo task_info;
    // task_type text,
    // table_name text,
    // sample_size text,
    // col_name text,
    // dataset_name text,
    // select_model_path text,
    // regression_model_path text
    char* task_type = (char*)list_nth(inputs, 0);
    TaskType task_type_enum;
    if (strcmp(task_type, "image_classification") == 0) {
        task_type_enum = TaskType::IMAGE_CLASSIFICATION;
    } else if (strcmp(task_type, "predict") == 0) {
        task_type_enum = TaskType::PREDICT;
    } else {
        ereport(ERROR, (errmsg("unknown task type %s", task_type)));
        return AgentAction::FAILURE;
    }
    task_info.task_type = task_type_enum;
    task_info.table_name = (char*)list_nth(inputs, 1);
    task_info.limit_length = atoi((char*)list_nth(inputs, 2));
    if (task_info.task_type == TaskType::IMAGE_CLASSIFICATION) {
        task_info.select_table_name = (char*)list_nth(inputs, 3);
        task_info.sample_size = atoi((char*)list_nth(inputs, 4));
        task_info.col_name = (char*)list_nth(inputs, 5);
        task_info.dataset_name = (char*)list_nth(inputs, 6);
        task_info.select_model_path = (char*)list_nth(inputs, 7);
        task_info.regression_model_path = (char*)list_nth(inputs, 8);
    }
    state->task_info.emplace_back(task_info);
    state->last_action = AgentAction::PERCEPTION;
    return AgentAction::SCHEDULE;
}

// orchestration agent: model selection, resource management
AgentAction OrchestrationAgent::Execute(std::shared_ptr<AgentState> state) {
    TaskInit(state);
    SPIRegisterProcess();
    // to be done
    state->last_action = AgentAction::ORCHESTRATION;
    return AgentAction::SCHEDULE;
}

void OrchestrationAgent::SPIRegisterProcess() {
    register_callback();
    return;
}

void OrchestrationAgent::TaskInit(std::shared_ptr<AgentState> state) {
    state->current_start_index = 0;
    state->current_end_index = 0;
    for (auto& task_unit : state->task_info) {
        // 先默认窗口为10
        int window_size = 10;
        char* task_model;
        char* task_cuda;
        switch (task_unit.task_type) {
            case TaskType::IMAGE_CLASSIFICATION:
                {
                    std::string selected_model = SelectModel(state, task_unit.select_table_name, task_unit.col_name, task_unit.sample_size, task_unit.dataset_name, task_unit.select_model_path, task_unit.regression_model_path);
                    task_model = const_cast<char*>(selected_model.c_str());
                    task_cuda = const_cast<char*>("gpu");
                }
                break;
            case TaskType::PREDICT:
                {
                    if (strcmp(task_unit.table_name, "slice_test") == 0) {
                        window_size = 32;
                        task_model = const_cast<char*>("slice");
                        task_cuda = const_cast<char*>("cpu");
                    }
                }
                break;
            default:
                elog(ERROR, "unknown task type %d", task_unit.task_type);
                break;
        }
        for (int i = 0; i < (task_unit.limit_length - 1) / window_size + 1; i++) {
            Task* task = (Task*)palloc0(sizeof(Task));
            if (task == NULL) {
                elog(ERROR, "Failed to allocate memory for Task");
                throw std::bad_alloc();
            }
            task->model = const_cast<char*>(task_model);
            task->cuda = const_cast<char*>(task_cuda);
            task->table_name = task_unit.table_name;
            // task->input_start_index = (i < window_size) ? 0 : (i - window_size);
            // task->input_end_index = i + 1; // 包含start，不包含end
            task->input_start_index = i * window_size;
            task->input_end_index = ((i + 1) * window_size < task_unit.limit_length) ? (i + 1) * window_size : task_unit.limit_length; // 包含start，不包含end
            task->output_start_index = 0;
            task->output_end_index = task->input_end_index - task->input_start_index;
            state->task_list = lappend(state->task_list, task);
        }
    }
    elog(INFO, "task_list length: %d", list_length(state->task_list));
    for (int i = 0; i < list_length(state->task_list); i++) {
        Task* task = (Task*)list_nth(state->task_list, i);
        elog(INFO, "task %d: model: %s, cuda: %s, table_name: %s, input_start_index: %ld, input_end_index: %ld, output_start_index: %ld, output_end_index: %ld", i, task->model, task->cuda, task->table_name, task->input_start_index, task->input_end_index, task->output_start_index, task->output_end_index);
    }
}

std::string OrchestrationAgent::SelectModel(std::shared_ptr<AgentState> state, const std::string& select_table_name, const std::string& col_name, int sample_size, const std::string& dataset_name, const std::string& select_model_path, const std::string& regression_model_path) {
    ModelSelection image_classification(select_model_path, regression_model_path);
    std::string temp_result = image_classification.SelectModel(select_table_name, col_name, sample_size, dataset_name);
    size_t pos = temp_result.find('%');
    if (pos == std::string::npos) {
        // 没找到 %, 根据需求处理
        throw std::runtime_error("SelectModel: invalid result string");
    }
    if (select_table_name.size() < 12) {
        throw std::runtime_error("SelectModel: invalid select_table_name");
    }
    std::string arch = temp_result.substr(0, pos);
    std::string pretrain_ds = temp_result.substr(pos + 1);
    std::string finetune_ds = dataset_name;
    std::string result_str = arch + "_" + arch + "_" + finetune_ds + "_" + "from" + "_" + pretrain_ds;
    return result_str;
}

// optimization agent: planning tree optimization
AgentAction OptimizationAgent::Execute(std::shared_ptr<AgentState> state) {
    // to be done 
    state->last_action = AgentAction::OPTIMIZATION;
    return AgentAction::SCHEDULE;
}

// execution agent: execute the plan
// clear all the tasks
AgentAction ExecutionAgent::Execute(std::shared_ptr<AgentState> state) {
    Task* task = (Task*)list_nth(state->task_list, state->current_task_id);
    elog(INFO, "Execution task: %s, %s, %s, %ld, %ld", task->model, task->cuda, task->table_name, task->input_start_index, task->input_end_index);
    if (task->input_start_index >= task->input_end_index) {
        ereport(ERROR, (errmsg("task input range error")));
        return AgentAction::FAILURE;
    }
    state->current_state.model = task->model;
    state->current_state.cuda = task->cuda;
    // 准备上下文
    // TODO: 缓冲区仅支持一种任务，可通过设置多类型缓冲区扩展
    while (state->current_start_index < task->input_start_index) {
        state->current_state.ins = list_delete_first(state->current_state.ins);
        // 内部逻辑似乎是覆盖而非清理
        // state->current_state.outs = list_delete_first(state->current_state.outs);
        state->current_start_index++;
    }
    while (state->current_end_index < task->input_end_index) {
        Args* vec = MemoryManager::LoadOneRow(task->table_name, state->current_end_index);
        state->current_state.ins = lappend(state->current_state.ins, vec);
        state->current_end_index++;
    }
    infer_batch_internal(&state->current_state, true);
    // Args* res = MemoryManager::LoadOneRow("slice_test", 0);
    // state->current_state.ins = lappend(state->current_state.ins, res);
    // to be done
    state->last_action = AgentAction::EXECUTION;
    return AgentAction::SCHEDULE;
}

// evaluation agent: evaluate the execution result
AgentAction EvaluationAgent::Execute(std::shared_ptr<AgentState> state) {
    // TODO: 目前逻辑Execution和Evaluation强相关（current_state），考虑将其解耦
    Task* task = (Task*)list_nth(state->task_list, state->current_task_id);
    for (int i = task->output_start_index; i < task->output_end_index; i++) {
        Args* ret = (Args*)list_nth(state->current_state.outs, i);
        elog(INFO, "Evaluation Result on index %d: %f", task->input_start_index + i, ret->floating);
    }
    // to be done
    state->last_action = AgentAction::EVALUATION;
    return AgentAction::SCHEDULE;
}

AgentAction ScheduleAgent::Execute(std::shared_ptr<AgentState> state) {
    // to be done
    try {
        if (state->last_action == AgentAction::START) {
            return AgentAction::PERCEPTION;
        } else if (state->last_action == AgentAction::PERCEPTION) {
            return AgentAction::ORCHESTRATION;
        } else if (state->last_action == AgentAction::ORCHESTRATION) {
            return AgentAction::OPTIMIZATION;
        } else if (state->last_action == AgentAction::OPTIMIZATION) {
            // to be done: 决策执行什么任务，若无任务可执行应当直接调到结束状态
            if (list_length(state->task_list) == 0) {
                elog(INFO, "ScheduleAgent: no task left, exit");
                return AgentAction::SUCCESS;
            }
            state->current_task_id = 0;
            return AgentAction::EXECUTION;
        } else if (state->last_action == AgentAction::EXECUTION) {
            return AgentAction::EVALUATION;
        } else if (state->last_action == AgentAction::EVALUATION) {
            // to be done: 如果可以，将已执行的任务删除
            if (list_length(state->task_list) > state->current_task_id + 1) {
                state->current_task_id++;
                return AgentAction::EXECUTION;
            } else {
                return AgentAction::SUCCESS;
            } 
        } else {
            return AgentAction::SUCCESS;
        }
    } catch (const std::exception& e) {
        elog(INFO, "ScheduleAgent error message:%s", e.what());
        return AgentAction::FAILURE;
    }
}