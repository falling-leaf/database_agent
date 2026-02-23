#include "model_agent.h"
#include "model_manager.h"
#include "model_utils.h"
#include "vector.h"
#include "embedding.h"
#include "spi_connection.h"
#include "md5.h"
#include "cpu_load_predictor.h"
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <array>
#include <memory>
#include <fstream>
#include <regex>
#include <sstream>
#include <algorithm>

#define ONLY_FOR_IMAGE_PREDICT true
#define ADD_OPTIMIZATION_LOGIC true
#define CPU_SAMPLE_BUTTON false
#define GPU_SAMPLE_BUTTON false
#define WINDOW_SIZE 32
#define SAMPLE_SIZE 10

extern "C" {
#include "catalog/pg_type_d.h"
#include "access/htup.h"
#include "access/tupdesc.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "miscadmin.h"

extern void infer_batch_internal(VecAggState *state, bool ret_float8);
}

extern void register_callback();
extern ModelManager model_manager;
extern MemoryManager memory_manager;

// Global CPU load predictor instance
static CPULoadPredictor cpu_load_predictor;

// Global GPU load predictor instance
static GPULoadPredictor gpu_load_predictor;

// Global variables to track ExecutionAgent timing
static std::chrono::high_resolution_clock::time_point execution_start_time;
static bool execution_timing_active = false;
static bool data_cleaning = false;

// Function to start ExecutionAgent timing
void StartExecutionAgentTiming() {
    execution_start_time = std::chrono::high_resolution_clock::now();
    execution_timing_active = true;
}

// Function to get ExecutionAgent runtime in microseconds
long long GetExecutionAgentRuntimeUs() {
    if (!execution_timing_active) {
        return 0; // Return 0 if timing hasn't started
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - execution_start_time);
    execution_timing_active = false; // Reset timing flag
    return duration.count();
}

// Implementation of the timed execution wrapper
AgentAction BaseAgentNode::ExecuteWithTiming(std::shared_ptr<AgentState> state) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Execute the actual agent logic
    AgentAction result = this->Execute(state);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Get the agent name for tracking
    std::string agent_name = this->Name();
    
    // Update the accumulated execution time in memory_manager
    memory_manager.execution_time_map[agent_name] += duration.count();
    memory_manager.execution_count_map[agent_name]++;
    
    return result;
}

void MemoryManager::PrintTimingStats() {
    elog(INFO, "=== Agent Execution Timing Statistics ===");
    double overall_total_time_ms = 0;
    for (const auto& pair : execution_time_map) {
        std::string agent_name = pair.first;
        long long total_time_microseconds = pair.second;  // time in microseconds
        int call_count = execution_count_map[agent_name];
        
        // Convert to milliseconds with nanosecond precision (as floating point milliseconds)
        double total_time_ms = total_time_microseconds / 1000.0;
        double avg_time_ms = call_count > 0 ? total_time_microseconds / (1000.0 * call_count) : 0.0;
        overall_total_time_ms += total_time_ms;
        
        elog(INFO, "Agent: %-20s Total: %10.3f ms, Calls: %6d, Average: %10.6f ms", 
             agent_name.c_str(), total_time_ms, call_count, avg_time_ms);
    }
    elog(INFO, "Overall Total: %10.3f ms", overall_total_time_ms);
    elog(INFO, "=========================================");
}

Args* MemoryManager::Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim)
{
    int mvec_oid = 0;
    // get_mvec_oid定义于model_uitls.cpp, 用于获取mvec类型的oid
    if (!get_mvec_oid(mvec_oid))
    {
        ereport(ERROR, (errmsg("get mvec oid error!")));
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
            MVec* cur_mvec = DatumGetMVec(datum);
            vec[i - start].ptr = cur_mvec;
            // elog(INFO, "found mvec; ref is: %d, dim is: %d, shape_size is: %d", cur_mvec->ref_d.is_ref_tag, cur_mvec->vec_d.dim, cur_mvec->vec_d.shape_size);
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
    Args* vec = NULL;
    // elog(INFO, "start converting row to vector, ncols=%d", ncols);
    if (table_name == "cifar_image_vector_table") {
        vec = MemoryManager::Tuple2Vec(tuple, tupdesc, 1, 1);
    }
    else {
        vec = MemoryManager::Tuple2Vec(tuple, tupdesc, 0, 1);
    }

    // elog(INFO, "finished converting row to vector");
    return vec;
}

void PerceptionAgent::LoadMVecData(MVec* current_data) {
    if (IS_MVEC_REF(current_data)) {
        int cache_idx = memory_manager.current_func_call;
        memory_manager.ins_cache[cache_idx] = (MVec*)palloc0(sizeof(Args));
        memory_manager.ins_cache[cache_idx] = current_data;
        return;
        // ereport(ERROR, (errmsg("Input vector is a Reference (RowId: %ld), but api_agent requires materialized data. "
        //                        "Please use de-referencing functions or ensure input is a concrete vector.", 
        //                        (long)GET_MVEC_ROWID(current_data))));
    }

    // 2. 安全地获取维度
    int dim = current_data->vec_d.dim;

    // 【关键检查 2】维度合法性校验 (MAX_VECTOR_DIM 定义在 vector.h 中，通常为 1亿左右)
    if (dim <= 0 || dim > MAX_VECTOR_DIM) {
        ereport(ERROR, (errmsg("Invalid vector dimension: %d. Expected between 1 and %d.", 
                               dim, MAX_VECTOR_DIM)));
    }

    // 3. 准备内存上下文
    int cache_idx = memory_manager.current_func_call;
    MemoryContext old_context = MemoryContextSwitchTo(TopMemoryContext);

    // 4. 计算大小并分配 (Header + Data)
    // 此时 dim 已经被校验过，不会导致 6GB 的请求
    size_t total_size = MVEC_HEADER_SIZE + ((size_t)dim * sizeof(float));
    
    // 确保 ins_cache 已初始化 (防御性编程)
    if (memory_manager.ins_cache == NULL) {
         // 这里可以补救或者报错，视你的初始化逻辑而定
         ereport(ERROR, (errmsg("Global memory ins_cache is not initialized.")));
    }

    memory_manager.ins_cache[cache_idx] = (MVec*)palloc0(total_size);

    // 5. 拷贝 Header
    std::memcpy(memory_manager.ins_cache[cache_idx], current_data, MVEC_HEADER_SIZE);

    // 6. 获取数据区指针并拷贝数据
    // 注意：目标地址紧跟在 Header 之后
    float* data_dst = (float*)((char*)memory_manager.ins_cache[cache_idx] + MVEC_HEADER_SIZE);
    
    // 从源数据的 data 数组起始处拷贝
    // 注意：current_data->vec_d.data 是柔性数组，直接指向 Header 后的内存
    std::memcpy(data_dst, current_data->vec_d.data, sizeof(float) * dim);

    // 7. 设置辅助指针
    memory_manager.ins_cache_data[cache_idx] = data_dst;

    MemoryContextSwitchTo(old_context);
}

char* str_to_lower(const char* str) {
    if (str == nullptr) {
        return nullptr;
    }
    // 分配内存存储小写字符串（+1 留空字符位置）
    char* lower_str = (char*)malloc(strlen(str) + 1);
    if (lower_str == nullptr) {
        elog(ERROR, "Failed to allocate memory for lower case string");
        return nullptr;
    }
    
    // 逐个字符转换为小写
    for (int i = 0; str[i] != '\0'; i++) {
        lower_str[i] = tolower((unsigned char)str[i]);
    }
    // 末尾添加字符串结束符
    lower_str[strlen(str)] = '\0';
    
    return lower_str;
}

AgentAction PerceptionAgent::Execute(std::shared_ptr<AgentState> state) {
    if (memory_manager.current_func_call == -1) {
        if (memory_manager.ins_cache == NULL)
        {
            elog(INFO, "reset the space");
            MemoryContext old_context = MemoryContextSwitchTo(TopMemoryContext);
            memory_manager.ins_cache_data = (float**)palloc0(sizeof(float*) * MAX_CACHE_SIZE);
            memory_manager.ins_cache = (MVec**)palloc0(sizeof(MVec*) * MAX_CACHE_SIZE);
            memory_manager.out_cache_data = (float*)palloc0(sizeof(float) * MAX_CACHE_SIZE);
            memory_manager.ins_buffer = (Args*)palloc0(WINDOW_SIZE * sizeof(Args));
            MemoryContextSwitchTo(old_context);
            
            elog(NOTICE, "Global memory allocated in TopMemoryContext.");
        }
        memory_manager.current_func_call = 0;
    } else {
        memory_manager.current_func_call++;
    }
    FunctionCallInfo fcinfo = state->fcinfo;
    // elog(INFO, "the number of param: %d", PG_NARGS());
    for (int i = 1; i < PG_NARGS(); i++) {
        if (PG_ARGISNULL(i)) {
            elog(INFO, "param %d is null", i);
            throw std::runtime_error("NULL param accepted.");
        }
    }
    TaskInfo task_info;
    int load_index = 1;
    char* task_type = (char*)PG_GETARG_CSTRING(load_index++);
    TaskType task_type_enum;
    if (strcmp(task_type, "image_classification") == 0) {
        task_type_enum = TaskType::IMAGE_CLASSIFICATION;
    } else if (strcmp(task_type, "series") == 0) {
        task_type_enum = TaskType::SERIES;
    } else if (strcmp(task_type, "nlp") == 0) {
        task_type_enum = TaskType::NLP;
    } else {
        ereport(ERROR, (errmsg("unknown task type %s", task_type)));
        return AgentAction::FAILURE;
    }
    task_info.task_type = task_type_enum;
    if (memory_manager.current_func_call == 0) {
        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
        state->task_info.emplace_back(task_info);
        MemoryContextSwitchTo(old_ctx);
    }
    // 下面代码每次调用均会执行
    MVec* current_data;
    if (task_info.task_type == TaskType::IMAGE_CLASSIFICATION) {
        if (ONLY_FOR_IMAGE_PREDICT) {
            current_data = (MVec*)PG_GETARG_MVEC_P(load_index++);
        } else {
            char* image_path = text_to_cstring(PG_GETARG_TEXT_PP(load_index++));
            if (memory_manager.sample_path.size() <= SAMPLE_SIZE)
                memory_manager.sample_path.emplace_back(image_path);
            char* check_result = str_to_lower(image_path);
            if (strstr(check_result, "cifar10") != nullptr) {
                // elog(INFO, "cifar10 image path: %s", image_path);
                current_data = image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, image_path);
            } else if (strstr(check_result, "stanford_dogs") != nullptr) {
                // elog(INFO, "stanford_dogs image path: %s", image_path);
                current_data = image_to_vector(256,224,0.485,0.456,0.406,0.229,0.224,0.225, image_path);
            } else {
                // elog(INFO, "default image path: %s", image_path);
                current_data = image_to_vector(224,224,0.4914,0.4822,0.4465,0.2023,0.1994,0.2010, image_path);
            }
        }
    }
    else current_data = (MVec*)PG_GETARG_MVEC_P(load_index++);
    LoadMVecData(current_data);
    // 打印验证：指针与部分数据
    // elog(INFO, "ins_cache[%d] header_size=%zu dim=%d data_ptr=%p first=%f", cache_idx, header_size, dim, (void*)(*data_slot), memory_manager.ins_cache_data[cache_idx][0]);
    // elog(NOTICE, "Global memory allocated in TopMemoryContext.");
    state->last_action = AgentAction::PERCEPTION;
    return AgentAction::SCHEDULE;
}

// orchestration agent: model selection, resource management
AgentAction OrchestrationAgent::Execute(std::shared_ptr<AgentState> state) {
    TaskInit(state);
    if (memory_manager.current_func_call == WINDOW_SIZE - 1 || (memory_manager.current_func_call < WINDOW_SIZE - 1 && memory_manager.is_last_call == 1))
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
    for (auto& task_unit : state->task_info) {
        int window_size = 32;
        std::string selected_model;
        bool ready_for_task = (memory_manager.current_func_call % WINDOW_SIZE == WINDOW_SIZE - 1) ||
                              (memory_manager.is_last_call == 1);
        if (!ready_for_task)
            continue;
        bool from_select_model = false;
        if (memory_manager.current_func_call == WINDOW_SIZE - 1) {
            elog(INFO, "start setting model and cuda");
            switch (task_unit.task_type) {
                case TaskType::IMAGE_CLASSIFICATION:
                    {
                        selected_model = SelectModel(state, task_unit.task_type);
                        if (!ONLY_FOR_IMAGE_PREDICT)
                            from_select_model = true;
                        // 关键修复：使用 pstrdup 复制字符串到 PostgreSQL 内存上下文
                        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                        task_unit.model_name = pstrdup(selected_model.c_str());
                        task_unit.cuda_name = pstrdup("gpu");   
                        MemoryContextSwitchTo(old_ctx);
                    }
                    break;
                case TaskType::SERIES:
                    {
                        selected_model = SelectModel(state, task_unit.task_type);
                        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                        task_unit.model_name = pstrdup(selected_model.c_str());
                        task_unit.cuda_name = pstrdup("cpu");
                        MemoryContextSwitchTo(old_ctx);
                    }
                    break;
                case TaskType::NLP:
                    {
                        selected_model = SelectModel(state, task_unit.task_type);
                        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                        task_unit.model_name = pstrdup(selected_model.c_str());
                        task_unit.cuda_name = pstrdup("gpu");
                        MemoryContextSwitchTo(old_ctx);
                    }
                    break;
                default:
                    elog(ERROR, "unknown task type %d", task_unit.task_type);
                    break;
            }
            if (!ONLY_FOR_IMAGE_PREDICT) {
                if (InitModel(selected_model.c_str(), from_select_model)) {
                elog(INFO, "InitModel success");
                }
                else {
                    elog(ERROR, "InitModel failed");
                }
            }
        }
        int64_t window_start = (memory_manager.current_func_call / WINDOW_SIZE) * WINDOW_SIZE;
        int64_t window_end = window_start + WINDOW_SIZE;
        if (memory_manager.is_last_call == 1) {
            window_end = memory_manager.current_func_call + 1;
        }
        Task* task = (Task*)palloc0(sizeof(Task));
        if (task == NULL) {
            elog(ERROR, "Failed to allocate memory for Task");
            throw std::bad_alloc();
        }
        // 关键修复：为每个 task 复制字符串，而不是共享同一个指针
        task->model = pstrdup(task_unit.model_name);
        task->cuda = pstrdup(task_unit.cuda_name);
        task->input_start_index = window_start;
        task->input_end_index = window_end; // 包含start，不包含end
        task->output_start_index = 0;
        task->output_end_index = task->input_end_index - task->input_start_index;
        state->task_list = lappend(state->task_list, task);
    }
    // elog(INFO, "task_list length: %d", list_length(state->task_list));
    // for (int i = 0; i < list_length(state->task_list); i++) {
    //     Task* task = (Task*)list_nth(state->task_list, i);
    //     elog(INFO, "task %d: model: %s, cuda: %s, input_start_index: %ld, input_end_index: %ld, output_start_index: %ld, output_end_index: %ld", i, task->model, task->cuda, task->input_start_index, task->input_end_index, task->output_start_index, task->output_end_index);
    // }
}

std::string OrchestrationAgent::SelectModel(std::shared_ptr<AgentState> state, TaskType task_type) {
    int sample_size = SAMPLE_SIZE;
    bool is_imagenet = false;
    if (task_type == TaskType::NLP) {
        MVec* input = memory_manager.ins_cache[0];
        int shape_size = GET_MVEC_SHAPE_SIZE(input);
        if (shape_size == 3) {
            int result0 = GET_MVEC_SHAPE_VAL(input, 0);
            int result1 = GET_MVEC_SHAPE_VAL(input, 1);
            int result2 = GET_MVEC_SHAPE_VAL(input, 2);
            if (result0 == 1 && result1 == 4 && result2 == 128) {
                return "sst2_vec";
            } else if (result0 == 1 && result1 == 2 && result2 == 133) {
                return "finance";
            } else {
                throw std::runtime_error("SelectModel: invalid input shape");
            }
        } else {
            elog(ERROR, "input shape size not 2");
            throw std::runtime_error("SelectModel: invalid input shape");
        }
    } else if (task_type == TaskType::SERIES) {
        MVec* input = memory_manager.ins_cache[0];
        int shape_size = GET_MVEC_SHAPE_SIZE(input);
        if (shape_size == 2) {
            int result0 = GET_MVEC_SHAPE_VAL(input, 0);
            int result1 = GET_MVEC_SHAPE_VAL(input, 1);
            if (result0 == 1 && result1 == 384) {
                return "slice";
            } else if (result0 == 1 && result1 == 2400) {
                return "swarm";
            } else if (result0 == 1 && result1 == 90) {
                return "year_predict";
            } else {
                throw std::runtime_error("SelectModel: invalid input shape");
            }
        } else {
            elog(ERROR, "input shape size not 2");
            throw std::runtime_error("SelectModel: invalid input shape");
        }
    } else if (task_type == TaskType::IMAGE_CLASSIFICATION) {
        if (ONLY_FOR_IMAGE_PREDICT) {
            MVec* input = memory_manager.ins_cache[0];
            int shape_size = GET_MVEC_SHAPE_SIZE(input);
            if (shape_size == 4) {
                int result0 = GET_MVEC_SHAPE_VAL(input, 0);
                int result1 = GET_MVEC_SHAPE_VAL(input, 1);
                int result2 = GET_MVEC_SHAPE_VAL(input, 2);
                int result3 = GET_MVEC_SHAPE_VAL(input, 3);
                if (result0 == 1 && result1 == 3 && result2 == 224 && result3 == 224) {
                    if (GET_MVEC_VAL(input, 0) > -2) {
                        return "googlenet_cifar10";
                    } else {
                        return "defect_vec";
                    }
                } else if (result0 == 1 && result1 == 3 && result2 == 256 && result3 == 224) {
                    return "alexnet_stanford_dogs";
                } else {
                    elog(ERROR, "input shape not (1, 3, 224, 224)");
                    throw std::runtime_error("SelectModel: invalid input shape");
                }
            } else {
                elog(ERROR, "input shape size not 4");
                throw std::runtime_error("SelectModel: invalid input shape");
            }
                return "googlenet_cifar10";
        }
        std::string finetune_ds;
        MVec* input = memory_manager.ins_cache[0];
        int shape_size = GET_MVEC_SHAPE_SIZE(input);
        if (shape_size == 4) {
            int result0 = GET_MVEC_SHAPE_VAL(input, 0);
            int result1 = GET_MVEC_SHAPE_VAL(input, 1);
            int result2 = GET_MVEC_SHAPE_VAL(input, 2);
            int result3 = GET_MVEC_SHAPE_VAL(input, 3);
            if (result0 == 1 && result1 == 3 && result2 == 224 && result3 == 224) {
                // TODO: cifar10和imagenet shape一样
                char* sample_path_lower = str_to_lower(memory_manager.sample_path[0].c_str());
                if (strstr(sample_path_lower, "cifar10")) {
                    finetune_ds = "cifar10";
                } else if (strstr(sample_path_lower, "image-net")) {
                    is_imagenet = true;
                    finetune_ds = "imagenet";
                } else {
                    elog(ERROR, "input shape not (1, 3, 224, 224)");
                    throw std::runtime_error("SelectModel: invalid input shape");
                }
            } else if (result0 == 1 && result1 == 3 && result2 == 256 && result3 == 224) {
                finetune_ds = "stanford_dogs";
            } else {
                elog(ERROR, "input shape not (1, 3, 224, 224)");
                throw std::runtime_error("SelectModel: invalid input shape");
            }
        } else {
            elog(ERROR, "input shape size not 4");
            throw std::runtime_error("SelectModel: invalid input shape");
        }
        // std::string select_model_path = "/home/why/pgdl/model/models/select_model/ViT-L-14_visual_traced.pt";
        std::string select_model_path = "/home/why/save/vit_l14_gpu_fixed.pt";
        std::string regression_model_path = "/home/why/pgdl/model/models/select_model/regression_model.onnx";
        ModelSelection image_classification(select_model_path, regression_model_path);

        std::string col_name = "padding";
        std::string temp_result = image_classification.SelectModel(col_name, col_name, sample_size, finetune_ds, memory_manager.sample_path);
        size_t pos = temp_result.find('%');
        if (pos == std::string::npos) {
            // 没找到 %, 根据需求处理
            throw std::runtime_error("SelectModel: invalid result string");
        }
        std::string arch = temp_result.substr(0, pos);
        std::string pretrain_ds = temp_result.substr(pos + 1);
        std::string result_str = arch + "_" + arch + "_" + finetune_ds + "_" + "from" + "_" + pretrain_ds;
        // elog(INFO, "SelectModel: %s", result_str.c_str());
        if (is_imagenet) {
            result_str = "resnet18_resnet18_imagenet";
        }
        return result_str;
    } else {
        throw std::runtime_error("SelectModel: invalid task type");
    }
}

// Function to analyze model characteristics: MAC, parameter count, and parameter size
ModelAnalysisResult AnalyzeModelWithInference(const std::string& model_name) {
    ModelAnalysisResult result = {0, 0, 0};
    
    std::string model_path;
    if (!model_manager.GetModelPath(model_name, model_path)) {
        ereport(WARNING, (errmsg("Model %s not found", model_name.c_str())));
        return result;
    }
    
    // Use the new AnalyzeModel method
    if (!model_manager.AnalyzeModel(model_name, model_path, result.mac_count, result.param_count, result.param_size_bytes)) {
        ereport(WARNING, (errmsg("Failed to analyze model %s", model_name.c_str())));
        return result;
    }
    
    return result;
}

bool OrchestrationAgent::InitModel(const char* model_name, bool from_select_model) {
    elog(INFO, "here");
    std::string full_path;
    if (from_select_model) {
        full_path = "/home/why/models_all/";
    } else {
        full_path = "/home/why/pgdl/model/models/";
    }
    // TODO: sentiment模型很大，MD5计算缓慢（硬性），因此时间计算时要把InitModel模块剥离出来
    if (strcmp(model_name, "sst2_vec") == 0) {
        full_path += "traced_albert_vec";
    } else if (strcmp(model_name, "finance") == 0) {
        full_path += "sentiment_analysis_model";
    } else {
        full_path += model_name;
    }
    full_path += ".pt";
    const char* model_path = full_path.c_str();
    const char* base_model_name = "";
    const char* discription = "";
    std::string base_model_path;
    int layer_size = 0;
    int mvec_oid = 0;
    ModelLayer* parameter_list = NULL;
    if(strlen(model_name) == 0){
        ereport(ERROR, (errmsg("model_name is empty!")));
    }
    
    if(access(model_path, F_OK) != 0){
        elog(INFO, "model_path: %s", model_path);
        ereport(ERROR, (errmsg("model is not exist!")));
    }

    model_manager.DropModel(model_name);
    if(strlen(base_model_name) == 0){
        if(model_manager.CreateModel(model_name, model_path, base_model_name, discription)){
            return true;
        }else {
            ereport(ERROR, (errmsg("create model error!")));
        }
    }else{
        if(!model_manager.IsBaseModelExist(base_model_name)){
            ereport(ERROR, (errmsg("base_model:%s not exist!", base_model_name)));
        }
        if(!model_manager.GetBaseModelPathFromBaseModel(base_model_name, base_model_path)){
            ereport(ERROR, (errmsg("base_model:%s not exist!", base_model_name)));
        }
        int ret = compare_model_struct(model_path, base_model_path.c_str());
        if(ret != 0){
            ereport(ERROR, (errmsg("model struct is not equal, errcode:%d", ret)));
        }
        ereport(INFO, (errmsg("model struct equals")));

        model_parameter_extraction(model_path, base_model_path.c_str(), &parameter_list, layer_size);

        ereport(INFO, (errmsg("model extraction success")));

        if(!get_mvec_oid(mvec_oid)){
            ereport(ERROR, (errmsg("get_mvec_oid error!")));
        }

        for(int i = 0; i<layer_size; i++){
            if(!insert_model_layer_parameter(model_name, parameter_list[i].layer_name, i+1, mvec_oid, parameter_list[i].layer_parameter)){
                ereport(ERROR, (errmsg("insert_model_layer_parameter error!")));
            }
        }

        ereport(INFO, (errmsg("insert parameter success")));

        for (int i = 0; i < layer_size; i++) {
            pfree(parameter_list[i].layer_name);
            pfree(parameter_list[i].layer_parameter);
        }
        if(parameter_list != NULL){
            pfree(parameter_list);
        }

        if(model_manager.CreateModel(model_name, model_path, base_model_name, discription)){
            return true;
        }else {
            ereport(ERROR, (errmsg("create model error!")));
        }
    }

    return false;
}

// Helper function to execute a command and return its output
std::string exec_command(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(static_cast<unsigned char>(*start))) {
        start++;
    }
    auto end = s.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(static_cast<unsigned char>(*end)));
    return std::string(start, end + 1);
}

// Function to collect GPU load and ExecutionAgent runtime data
void CollectGPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, ModelAnalysisResult analysis_result) {
    // Get GPU metrics
    double cuda_cores = 0.0;
    double gpu_freq = 0.0;
    double util = 0.0;
    double mem_used = 0.0;
    double mem_total = 0.0;
    
    try {
        // Get GPU status information
        std::string gpu_util_str = exec_command("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -c 100");
        util = std::stod(trim(gpu_util_str));
        
        std::string gpu_mem_used_str = exec_command("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -c 100");
        mem_used = std::stod(trim(gpu_mem_used_str));
        
        std::string gpu_mem_total_str = exec_command("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -c 100");
        mem_total = std::stod(trim(gpu_mem_total_str));
        
        // Get GPU FLOPS and frequency information
        std::string gpu_cap = exec_command("nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader,nounits | head -c 100");
        gpu_cap = trim(gpu_cap);
        
        if (gpu_cap == "8.0") {
            cuda_cores = 6912;
        } else if (gpu_cap == "7.5") {
            cuda_cores = 4608;
        } else if (gpu_cap == "7.0") {
            cuda_cores = 5120;
        } else if (gpu_cap == "8.6") {
            cuda_cores = 10496;
        } else {
            cuda_cores = 2048;
        }

        std::string gpu_clocks = exec_command("nvidia-smi --id=0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -c 100");
        gpu_freq = std::stod(trim(gpu_clocks));
    } catch (...) {
        // If GPU information is not available, set default values
        elog(INFO, "nvidia-smi command failed, set default values");
        cuda_cores = 0.0;
        gpu_freq = 0.0;
        util = 0.0;
        mem_used = 0.0;
        mem_total = 1.0;  // Avoid division by zero
    }
    
    // Get ExecutionAgent runtime measurement
    long long execution_runtime_us = GetExecutionAgentRuntimeUs();
    
    // Store the collected data to database via SPI
    SPIConnector spi_connector;
    if (!spi_connector.Connect()) {
        elog(WARNING, "Failed to connect to SPI for storing GPU load data");
        return;
    }
    
    // Insert the collected data
    // CREATE TABLE IF NOT EXISTS gpu_load_training_data (
    //     id SERIAL PRIMARY KEY,
    //     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    //     cuda_cores DOUBLE PRECISION,
    //     gpu_freq DOUBLE PRECISION,
    //     util DOUBLE PRECISION,
    //     mem_used DOUBLE PRECISION,
    //     mem_total DOUBLE PRECISION,
    //     execution_runtime_us BIGINT,
    //     data_shape_1 BIGINT,
    //     data_shape_2 BIGINT,
    //     data_shape_3 BIGINT,
    //     data_shape_4 BIGINT,
    //     model_mac_count BIGINT,
    //     model_param_count BIGINT,
    //     model_param_size BIGINT
    // );
    std::string insert_sql = "INSERT INTO gpu_load_training_data (cuda_cores, gpu_freq, util, mem_used, mem_total, execution_runtime_us, data_shape_1, data_shape_2, data_shape_3, data_shape_4, model_mac_count, model_param_count, model_param_size) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)";
    SPISqlWrapper sql_wrapper(spi_connector, insert_sql, 13);
    
    if (!sql_wrapper.Bind(1, FLOAT8OID, Float8GetDatum(cuda_cores)) ||
        !sql_wrapper.Bind(2, FLOAT8OID, Float8GetDatum(gpu_freq)) ||
        !sql_wrapper.Bind(3, FLOAT8OID, Float8GetDatum(util)) ||
        !sql_wrapper.Bind(4, FLOAT8OID, Float8GetDatum(mem_used)) ||
        !sql_wrapper.Bind(5, FLOAT8OID, Float8GetDatum(mem_total)) ||
        !sql_wrapper.Bind(6, INT8OID, Int64GetDatum(execution_runtime_us)) ||
        !sql_wrapper.Bind(7, INT8OID, Int64GetDatum(shape0)) ||
        !sql_wrapper.Bind(8, INT8OID, Int64GetDatum(shape1)) ||
        !sql_wrapper.Bind(9, INT8OID, Int64GetDatum(shape2)) ||
        !sql_wrapper.Bind(10, INT8OID, Int64GetDatum(shape3)) ||
        !sql_wrapper.Bind(11, INT8OID, Int64GetDatum(analysis_result.mac_count)) ||
        !sql_wrapper.Bind(12, INT8OID, Int64GetDatum(analysis_result.param_count)) ||
        !sql_wrapper.Bind(13, INT8OID, Int64GetDatum(analysis_result.param_size_bytes))) {
        elog(WARNING, "Failed to bind parameters for inserting GPU load data");
        return;
    }
    
    if (!sql_wrapper.Execute()) {
        elog(WARNING, "Failed to insert GPU load data into database");
        return;
    }
    
    elog(INFO, "Successfully stored GPU load data: cuda_cores=%.1f, freq=%.1f MHz, util=%.1f%%, mem_used=%.1f MiB, mem_total=%.1f MiB, runtime=%lld us, shapes=[%d,%d,%d,%d], model_mac=%lld, model_param=%lld, model_size=%zu", 
         cuda_cores, gpu_freq, util, mem_used, mem_total, execution_runtime_us, 
         shape0, shape1, shape2, shape3, 
         analysis_result.mac_count, analysis_result.param_count, analysis_result.param_size_bytes);
}

// Function to collect CPU load and ExecutionAgent runtime data
void CollectCPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, ModelAnalysisResult analysis_result) {
    // Get CPU load average
    double cpu_load = 0.0;
    int cpu_cores = 1;
    try {
        cpu_load = std::stod(exec_command("cat /proc/loadavg | awk '{print $1}'"));
        cpu_cores = std::stoi(exec_command("nproc"));
    } catch (...) { 
        cpu_load = 0.5; 
        cpu_cores = 1; 
    }
    
    // Get ExecutionAgent runtime measurement
    long long execution_runtime_us = GetExecutionAgentRuntimeUs();
    
    // Store the collected data to database via SPI
    SPIConnector spi_connector;
    if (!spi_connector.Connect()) {
        elog(WARNING, "Failed to connect to SPI for storing CPU load data");
        return;
    }
    // Calculate normalized load factor (similar to the current dynamic multiplier logic)
    double normalized_load_factor = (cpu_load / (double)cpu_cores) + 1.0;
    double threshold = 1.2 + 0.1 * std::log2((double)cpu_cores);
    
    if ((cpu_load / (double)cpu_cores) + 1.0 > threshold) {
        normalized_load_factor = ((cpu_load / (double)cpu_cores) + 1.0) * std::exp(5.0 * (((cpu_load / (double)cpu_cores) + 1.0) - threshold));
    }
    
    // Insert the collected data
    // CREATE TABLE IF NOT EXISTS cpu_load_training_data (
    //     id SERIAL PRIMARY KEY,
    //     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    //     cpu_load DOUBLE PRECISION,
    //     cpu_cores INTEGER,
    //     execution_runtime_us BIGINT,
    //     data_shape_1 BIGINT,
    //     data_shape_2 BIGINT,
    //     data_shape_3 BIGINT,
    //     data_shape_4 BIGINT,
    //     model_mac_count BIGINT,
    //     model_param_count BIGINT,
    //     model_param_size BIGINT
    //     )
    std::string insert_sql = "INSERT INTO cpu_load_training_data (cpu_load, cpu_cores, execution_runtime_us, data_shape_1, data_shape_2, data_shape_3, data_shape_4, model_mac_count, model_param_count, model_param_size) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)";
    SPISqlWrapper sql_wrapper(spi_connector, insert_sql, 10);
    
    if (!sql_wrapper.Bind(1, FLOAT8OID, Float8GetDatum(cpu_load)) ||
        !sql_wrapper.Bind(2, INT4OID, Int32GetDatum(cpu_cores)) ||
        !sql_wrapper.Bind(3, INT8OID, Int64GetDatum(execution_runtime_us)) ||
        !sql_wrapper.Bind(4, INT8OID, Int64GetDatum(shape0)) ||
        !sql_wrapper.Bind(5, INT8OID, Int64GetDatum(shape1)) ||
        !sql_wrapper.Bind(6, INT8OID, Int64GetDatum(shape2)) ||
        !sql_wrapper.Bind(7, INT8OID, Int64GetDatum(shape3)) ||
        !sql_wrapper.Bind(8, INT8OID, Int64GetDatum(analysis_result.mac_count)) ||
        !sql_wrapper.Bind(9, INT8OID, Int64GetDatum(analysis_result.param_count)) ||
        !sql_wrapper.Bind(10, INT8OID, Int64GetDatum(analysis_result.param_size_bytes))) {
        elog(WARNING, "Failed to bind parameters for inserting CPU load data");
        return;
    }
    
    if (!sql_wrapper.Execute()) {
        elog(WARNING, "Failed to insert CPU load data into database");
        return;
    }
    
    elog(INFO, "Successfully stored CPU load data: load=%.3f, cores=%d, runtime=%lld us, shapes=[%d,%d,%d,%d], model_mac=%lld, model_param=%lld, model_size=%zu", 
         cpu_load, cpu_cores, execution_runtime_us, shape0, shape1, shape2, shape3, 
         analysis_result.mac_count, analysis_result.param_count, analysis_result.param_size_bytes);
}

// Helper function to estimate CPU FLOPS
double estimate_cpu_flops() {
    try {
        // Read CPU info from /proc/cpuinfo
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        std::string model_name;
        double cpu_mhz = 0.0;
        int reading_state = 0;

        while (std::getline(cpuinfo, line) && reading_state < 2) {
            if (line.find("model name") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    model_name = line.substr(colon_pos + 2);
                    elog(INFO, "Detected CPU model: %s", model_name.c_str());
                    // 仅获取第一个CPU型号后继续循环，不立即跳出
                    reading_state += 1;
                }
            } else if (line.find("cpu MHz") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    cpu_mhz = std::stod(line.substr(colon_pos + 2));
                    elog(INFO, "Detected CPU frequency: %.2f MHz", cpu_mhz);
                    // 获取频率后也不跳出，继续读取直到文件结束
                    reading_state += 1;
                }
            }
        }
        
        // 用当前的实际频率进行估算
        double freq_ghz = cpu_mhz / 1000.0;
        
        double flops_per_cycle = 8.0;
        std::transform(model_name.begin(), model_name.end(), model_name.begin(), ::tolower);
        if (model_name.find("xeon") != std::string::npos) {
            flops_per_cycle = 32.0;  // AVX-512
        } else if (model_name.find("core") != std::string::npos || 
                   model_name.find("i3") != std::string::npos || 
                   model_name.find("i5") != std::string::npos || 
                   model_name.find("i7") != std::string::npos || 
                   model_name.find("i9") != std::string::npos) {
            flops_per_cycle = 16.0;  // AVX2
        }
        
        double result = flops_per_cycle * freq_ghz * 1e9;
        // elog(INFO, "Calculated CPU FLOPS: %.2e", result);
        return result;
    } catch (const std::exception& e) {
        elog(INFO, "Warning: Could not estimate CPU FLOPs, using default 50 GFLOPS. Error: %s", e.what());
        return 50e9; // Default value
    }
}

// Helper function to get GPU FLOPS
double get_gpu_flops() {
    try {
        std::string gpu_status = exec_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -c 5");
        if (gpu_status.empty()) {
            // 如果empty说明没有显卡驱动
            elog(INFO, "No GPU available, returning 0.0 FLOPS");
            return 0.0; // No GPU available
        }
        std::string gpu_cap = exec_command("nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader,nounits | head -c 100");
        gpu_cap = trim(gpu_cap);
        
        double cuda_cores = 0.0;
        if (gpu_cap == "8.0") {
            cuda_cores = 6912;
        } else if (gpu_cap == "7.5") {
            cuda_cores = 4608;
        } else if (gpu_cap == "7.0") {
            cuda_cores = 5120;
        } else if (gpu_cap == "8.6") {
            cuda_cores = 10496;
        } else {
            cuda_cores = 2048;
        }

        std::string gpu_clocks = exec_command("nvidia-smi --id=0 --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -c 100");
        double gpu_freq = std::stod(gpu_clocks);

        double result = cuda_cores * gpu_freq * 1e6 * 2.0;
        elog(INFO, "Calculated GPU FLOPS: %.2e", result);
        return result;
    } catch (const std::exception& e) {
        elog(INFO, "Warning: Could not get GPU FLOPs, using default 10 TFLOPS. Error: %s", e.what());
        return 10e12; // Default value
    }
}

// Helper function to get CPU memory bandwidth
double get_cpu_mem_bandwidth() {
    try {
        std::string mem_bandwidth_str = exec_command("echo \"morphingdbwhy\" | sudo -S dmidecode -t memory | grep Speed | head -n 100");
        std::vector<double> speeds;
        std::stringstream ss(mem_bandwidth_str);
        std::string line;
        while (std::getline(ss, line)) {
            if (line.find("Configured Memory Speed") != std::string::npos) {
                continue;
            }
            size_t pos = line.find("Speed:");
            if (pos != std::string::npos) {
                std::string speed_str = line.substr(pos + 6);
                speed_str.erase(std::remove(speed_str.begin(), speed_str.end(), ' '), speed_str.end());
                speeds.push_back(std::stod(speed_str));
            }
        }
        double avg_speed = std::accumulate(speeds.begin(), speeds.end(), 0.0) / speeds.size();
        elog(INFO, "Average memory bandwidth: %.2f MT/s", avg_speed);
        // TODO: 假设双通道64位总线
        double bandwidth_GBs = (avg_speed * 2 * 64 / 8) / 1000;
        return bandwidth_GBs * 1e9; // Convert to bytes/sec
    } catch (const std::exception& e) {
        elog(INFO, "Warning: Could not get memory bandwidth, using default 25.6 GB/s. Error: %s", e.what());
        return 25.6 * 1e9; // Default value
    }
}

// Helper function to get GPU memory bandwidth
double get_gpu_mem_bandwidth() {
    return 900 * 1e9;
    try {
        std::string bus_width_str = exec_command("nvidia-smi --query-gpu=memory_info.width --format=csv,noheader,nounits | head -c 10");

        std::string mem_clock_str = exec_command("nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits | head -c 10");

        if (!bus_width_str.empty() && !mem_clock_str.empty()) {
            int bus_width = std::stoi(bus_width_str);
            int mem_clock = std::stoi(mem_clock_str);

            // Calculate effective bandwidth (GDDR5/GDDR6 double data rate)
            double bandwidth = (bus_width * mem_clock * 2) / 8; // Convert to GB/s

            double result = bandwidth * 1e9; // Convert to bytes/sec
            elog(INFO, "Calculated GPU memory bandwidth: %.2e bytes/sec", result);
            return result;
        }
        elog(INFO, "Could not retrieve bus width or memory clock, using default bandwidth: %.2e", 900 * 1e9);
        return 900 * 1e9; // Default for RTX 3090
    } catch (const std::exception& e) {
        elog(INFO, "Warning: Could not get GPU bandwidth, using default 900 GB/s. Error: %s", e.what());
        return 900 * 1e9; // Default value
    }
}

// 获取系统平均负载 (1分钟内)
double get_cpu_load_factor() {
    try {
        std::string load_str = exec_command("cat /proc/loadavg | awk '{print $1}'");
        double load = std::stod(load_str);
        // 获取逻辑核心数
        std::string nproc_str = exec_command("nproc");
        double nproc = std::stod(nproc_str);
        // 如果负载超过核心数，说明存在严重的调度排队
        return (load / nproc); 
    } catch (...) { return 1.0; }
}

// 获取所有 GPU 的状态信息
std::vector<GpuStatus> get_all_gpu_status() {
    std::vector<GpuStatus> status_list;
    try {
        // 一次性查询所有 GPU 的 ID, 利用率, 已用显存, 总显存
        std::string output = exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits");
        std::stringstream ss(output);
        std::string line;
        while (std::getline(ss, line)) {
            GpuStatus gpu;
            char comma;
            std::stringstream ls(line);
            ls >> gpu.id >> comma >> gpu.util >> comma >> gpu.mem_used >> comma >> gpu.mem_total;
            status_list.push_back(gpu);
        }
    } catch (...) {}
    return status_list;
}

AgentAction OptimizationAgent::Execute(std::shared_ptr<AgentState> state) {
    // if (false) {
    if (ADD_OPTIMIZATION_LOGIC && (memory_manager.current_func_call % WINDOW_SIZE == WINDOW_SIZE - 1 || memory_manager.is_last_call == 1)) {
        bool enable_dynamic_cost = true; 
        if (enable_dynamic_cost) {
            elog(INFO, "OptimizationAgent::Execute - Using Dynamic Cost Model");

            // 1. 基础信息提取 (利用现有工具函数)
            double cpu_flops = estimate_cpu_flops();
            double cpu_bandwidth = get_cpu_mem_bandwidth();
            double gpu_bandwidth = get_gpu_mem_bandwidth();
            Task* task = (Task*)list_nth(state->task_list, 0);
            ModelAnalysisResult analysis_result = AnalyzeModelWithInference(task->model);
            int num_rows = WINDOW_SIZE;

            // --- CPU 动态 Cost 计算 ---
            // 静态部分
            double cpu_cost_static = (analysis_result.param_size_bytes / cpu_bandwidth) + 
                                    (analysis_result.mac_count / cpu_flops) * num_rows;
            
            // 动态部分：提取 CPU 负载情况 (Load Average)
            double cpu_load = 0.0;
            int cpu_cores = 1;
            try {
                cpu_load = std::stod(exec_command("cat /proc/loadavg | awk '{print $1}'"));
                cpu_cores = std::stoi(exec_command("nproc"));
            } catch (...) { cpu_load = 0.5; cpu_cores = 1; }
            
            // OLD FIXED FORMULA (preserved as comment):
            // 动态公平性修正：负载越高，调度延迟越大
            // 修正系数 = (当前负载 / 核心数) + 1.0 (基础开销)
            // double m = (cpu_load / (double)cpu_cores) + 1.0;

            // double threshold = 1.2 + 0.1 * std::log2((double)cpu_cores);
    
            // // 利用对数缩放模型进行辅助判断
            // double cpu_dynamic_multiplier = m;

            // // 3. 判定是否进入指数惩罚区
            // if (m > threshold) {
            //     // m * e^(m - T) 
            //     // 保证在 m = threshold 时曲线连续（e^0 = 1）
            //     // 且当 m 超过阈值后，Cost 会随负载比增加呈指数级飞升
            //     cpu_dynamic_multiplier = m * std::exp(5.0 * (m - threshold));
            // } else {
            //     cpu_dynamic_multiplier = m; // 线性增长区
            // }
            
            // NEW MODEL-BASED APPROACH:
            // Try to load the model from .pt file if not already loaded
            static bool model_tried_loading = false;
            if (!model_tried_loading) {
                if (!cpu_load_predictor.LoadModel("/home/why/cpu_load_model.pt")) {
                    elog(INFO, "Model file not found, attempting to train new model from database data");
                    if (cpu_load_predictor.TrainModel()) {
                        cpu_load_predictor.SaveModel("/home/why/cpu_load_model.pt");
                        elog(INFO, "New model trained and saved to /home/why/cpu_load_model.pt");
                    } else {
                        elog(WARNING, "Failed to train model, will use fallback logic");
                    }
                }
                model_tried_loading = true;
            }
            
            // Use the trained model to predict the dynamic multiplier
            MVec* input = memory_manager.ins_cache[0];
            int shape_size = GET_MVEC_SHAPE_SIZE(input);
            int result0 = 0, result1 = 0, result2 = 0, result3 = 0;
            if (shape_size == 4) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
                result2 = GET_MVEC_SHAPE_VAL(input, 2);
                result3 = GET_MVEC_SHAPE_VAL(input, 3);
            } else if (shape_size == 3) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
                result2 = GET_MVEC_SHAPE_VAL(input, 2);
            } else if (shape_size == 2) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
            }
            elog(INFO, "ModelAnalysisResult: %d, %d, %d", analysis_result.mac_count, analysis_result.param_count, analysis_result.param_size_bytes);
            std::vector<double> workload = {
                (double)result0,
                (double)result1,
                (double)result2,
                (double)result3,
                (double)analysis_result.mac_count,
                (double)analysis_result.param_count,
                (double)analysis_result.param_size_bytes
            };
            double cpu_dynamic_cost = cpu_load_predictor.Predict(workload, cpu_load, cpu_cores);
            
            elog(INFO, "Model-based CPU dynamic cost: %.3f (load=%.3f, cores=%d)", 
                 cpu_dynamic_cost, cpu_load, cpu_cores);

            double cpu_cost = cpu_dynamic_cost;

            // --- GPU 动态 Cost 计算与多卡评估 ---
            double min_gpu_cost = std::numeric_limits<double>::max();
            int best_gpu_id = -1;

            // 提取所有 GPU 状态：index, utilization, memory_used, memory_total
            std::string gpu_info = exec_command("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits");
            std::stringstream ss(gpu_info);
            std::string line;

            while (std::getline(ss, line)) {
                int idx;
                double util, mem_used, mem_total;
                char comma;
                std::stringstream ls(line);
                if (ls >> idx >> comma >> util >> comma >> mem_used >> comma >> mem_total) {
                    // 静态部分 (针对当前 GPU)
                    // 注意：此处假设各卡规格相近，使用全局 get_gpu_flops，可根据 idx 进一步精细化
                    double current_gpu_flops = get_gpu_flops(); 
                    double latency = 5e-4;
                    double gpu_cost_static = latency + (analysis_result.param_size_bytes / cpu_bandwidth) + 
                                            (analysis_result.param_size_bytes / gpu_bandwidth) + 
                                            (analysis_result.mac_count / current_gpu_flops) * num_rows;

                    // 使用 GPU 负载预测器获取动态乘数
                    std::vector<double> workload = {
                        (double)result0,
                        (double)result1,
                        (double)result2,
                        (double)result3,
                        (double)analysis_result.mac_count,
                        (double)analysis_result.param_count,
                        (double)analysis_result.param_size_bytes
                    };
                    
                    // 尝试加载或训练 GPU 负载预测模型
                    static bool gpu_model_tried_loading = false;
                    if (!gpu_model_tried_loading) {
                        if (!gpu_load_predictor.LoadModel("/home/why/gpu_load_model.pt")) {
                            elog(INFO, "GPU Model file not found, attempting to train new model from database data");
                            if (gpu_load_predictor.TrainModel()) {
                                gpu_load_predictor.SaveModel("/home/why/gpu_load_model.pt");
                                elog(INFO, "New GPU model trained and saved to /home/why/gpu_load_model.pt");
                            } else {
                                elog(WARNING, "Failed to train GPU model, will use fallback logic");
                            }
                        }
                        gpu_model_tried_loading = true;
                    }
                    
                    // 获取特定GPU的计算能力和频率信息
                    double cuda_cores = 0.0;
                    double gpu_freq = 0.0;
                    try {
                        // Get GPU compute capability for the specific GPU ID
                        std::string gpu_cap_cmd = "nvidia-smi --id=" + std::to_string(idx) + " --query-gpu=compute_cap --format=csv,noheader,nounits";
                        std::string gpu_cap = exec_command(gpu_cap_cmd.c_str());
                        gpu_cap = trim(gpu_cap);
                        
                        if (gpu_cap == "8.0") {
                            cuda_cores = 6912;
                        } else if (gpu_cap == "7.5") {
                            cuda_cores = 4608;
                        } else if (gpu_cap == "7.0") {
                            cuda_cores = 5120;
                        } else if (gpu_cap == "8.6") {
                            cuda_cores = 10496;
                        } else {
                            cuda_cores = 2048; // Default value
                        }

                        // Get GPU max SM clock frequency for the specific GPU ID
                        std::string gpu_freq_cmd = "nvidia-smi --id=" + std::to_string(idx) + " --query-gpu=clocks.max.sm --format=csv,noheader,nounits";
                        std::string gpu_clocks = exec_command(gpu_freq_cmd.c_str());
                        gpu_freq = std::stod(trim(gpu_clocks));
                    } catch (...) {
                        // If GPU information is not available for this specific GPU, set default values
                        cuda_cores = 2048;  // Default CUDA cores
                        gpu_freq = 1500.0;  // Default frequency in MHz
                    }
                    
                    // 构造GPU资源向量: {cuda_cores, gpu_freq, util, mem_used, mem_total}
                    std::vector<double> gpu_resource = {
                        cuda_cores,           // cuda_cores - 特定GPU的CUDA核心数
                        gpu_freq,             // gpu_freq - 特定GPU的频率
                        util,                 // util - 当前GPU利用率
                        mem_used,             // mem_used - 已使用的显存
                        mem_total             // mem_total - 总显存
                    };
                    
                    double predicted_runtime = gpu_load_predictor.Predict(workload, gpu_resource);
                    elog(INFO, "predicted_runtime: %.2e", predicted_runtime);

                    double current_gpu_total_cost = predicted_runtime;
                    // elog(INFO, "current_gpu_total_cost: %.2e", current_gpu_total_cost);
                    if (current_gpu_total_cost < min_gpu_cost) {
                        min_gpu_cost = current_gpu_total_cost;
                        best_gpu_id = idx;
                    }
                }
            }

            // 2. 最终决策
            double final_gpu_cost = (best_gpu_id == -1) ? std::numeric_limits<double>::max() : min_gpu_cost;
            elog(INFO, "cpu_cost: %.2e", cpu_cost);
            elog(INFO, "final_gpu_cost: %.2e", final_gpu_cost);
            if (final_gpu_cost < cpu_cost) {
                std::string gpu_str = "gpu:" + std::to_string(best_gpu_id);
                MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                task->cuda = pstrdup(gpu_str.c_str());
                MemoryContextSwitchTo(old_ctx);
                
                // 记录选择的 GPU ID 供后续调度使用
                elog(INFO, "Decision: Switch to GPU %d", best_gpu_id);
                // set_target_gpu_id(best_gpu_id); // 假设有对应设置函数
            } else {
                MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                task->cuda = pstrdup("cpu");
                MemoryContextSwitchTo(old_ctx);
                elog(INFO, "Decision: Switch to CPU");
            }

        } else if (false) {
            // --- 原有静态逻辑完全不变 ---
            elog(INFO, "OptimizationAgent::Execute - Starting hardware information gathering");
            // Get hardware information
            double cpu_flops = estimate_cpu_flops();
            double gpu_flops = get_gpu_flops();
            double cpu_bandwidth = get_cpu_mem_bandwidth();
            double gpu_bandwidth = get_gpu_mem_bandwidth();

            elog(INFO, "OptimizationAgent::Execute - CPU FLOPS retrieved: %.2e", cpu_flops);
            elog(INFO, "OptimizationAgent::Execute - GPU FLOPS retrieved: %.2e", gpu_flops);
            elog(INFO, "OptimizationAgent::Execute - CPU bandwidth retrieved: %.2e", cpu_bandwidth);
            elog(INFO, "OptimizationAgent::Execute - GPU bandwidth retrieved: %.2e", gpu_bandwidth);

            Task* task = (Task*)list_nth(state->task_list, 0);
            
            ModelAnalysisResult  analysis_result = AnalyzeModelWithInference(task->model);
            elog(INFO, "ModelAnalysisResult: %d, %d, %d", analysis_result.mac_count, analysis_result.param_count, analysis_result.param_size_bytes);
            int num_rows = WINDOW_SIZE;
            double latency = 1e-4;
            double gpu_cost = (analysis_result.param_size_bytes / cpu_bandwidth) + (analysis_result.param_size_bytes / gpu_bandwidth) + latency + (analysis_result.mac_count / gpu_flops) * num_rows;
            elog(INFO, "GPU cost: %.2e", gpu_cost);
            
            double cpu_cost = (analysis_result.param_size_bytes / cpu_bandwidth) + (analysis_result.mac_count / cpu_flops) * num_rows;
            elog(INFO, "CPU cost: %.2e", cpu_cost);

            if (gpu_cost < cpu_cost) {
                MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                task->cuda = pstrdup("gpu");
                MemoryContextSwitchTo(old_ctx);
            } else {
                MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                task->cuda = pstrdup("cpu");
                MemoryContextSwitchTo(old_ctx);
            }
        }
    }

    state->last_action = AgentAction::OPTIMIZATION;
    return AgentAction::SCHEDULE;
}

// execution agent: execute the plan
// clear all the tasks
AgentAction ExecutionAgent::Execute(std::shared_ptr<AgentState> state) {
    Task* task = (Task*)list_nth(state->task_list, state->current_task_id);
    if (CPU_SAMPLE_BUTTON) {
        task->cuda = "cpu";
    } else if (GPU_SAMPLE_BUTTON) {
        task->cuda = "gpu";
    }
    elog(INFO, "Execution task: %s, %s, %ld, %ld", task->model, task->cuda, task->input_start_index, task->input_end_index);
    
    if (task->input_start_index >= task->input_end_index) {
        ereport(ERROR, (errmsg("task input range error")));
        return AgentAction::FAILURE;
    }

    // 设置模型参数
    state->current_state.model = task->model;
    state->current_state.cuda = task->cuda;

    // 【关键修正 2】: 抛弃“滑动窗口”逻辑，每次执行 Task 前彻底重置 Input/Output 列表
    // 之前的逻辑导致 outs 列表无限增长，导致第二次 infer 时崩溃
    state->current_state.ins = NIL;
    state->current_state.outs = NIL;

    const int64_t count = task->input_end_index - task->input_start_index;
    for (int64_t i = 0; i < count; i++) {
        const int64_t current_index = task->input_start_index + i;
        Args* vec = memory_manager.ins_buffer + i; 
        if (memory_manager.ins_cache[current_index] == NULL) {
            ereport(ERROR, (errmsg("ins_cache is null at index %ld", current_index)));
        }
        vec->ptr = memory_manager.ins_cache[current_index];
        state->current_state.ins = lappend(state->current_state.ins, vec);
    }

    elog(INFO, "Constructed batch size: %d. Starting inference...", list_length(state->current_state.ins));

    // Start timing for ExecutionAgent
    if (ADD_OPTIMIZATION_LOGIC)
        StartExecutionAgentTiming();

    // 执行推理


    infer_batch_internal(&state->current_state, true);

    // Collect CPU load and runtime data after execution
    if (ADD_OPTIMIZATION_LOGIC && (list_length(state->current_state.ins) == WINDOW_SIZE)) {
        if (!data_cleaning)
            data_cleaning = true;
        else {
            MVec* input = memory_manager.ins_cache[0];
            int shape_size = GET_MVEC_SHAPE_SIZE(input);
            int result0 = 0, result1 = 0, result2 = 0, result3 = 0;
            if (shape_size == 4) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
                result2 = GET_MVEC_SHAPE_VAL(input, 2);
                result3 = GET_MVEC_SHAPE_VAL(input, 3);
            } else if (shape_size == 3) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
                result2 = GET_MVEC_SHAPE_VAL(input, 2);
            } else if (shape_size == 2) {
                result0 = GET_MVEC_SHAPE_VAL(input, 0);
                result1 = GET_MVEC_SHAPE_VAL(input, 1);
            }
            ModelAnalysisResult  analysis_result = AnalyzeModelWithInference(task->model);
            elog(INFO, "ModelAnalysisResult: %d, %d, %d", analysis_result.mac_count, analysis_result.param_count, analysis_result.param_size_bytes);
            if (task->cuda == NULL || strcmp(task->cuda, "cpu") == 0)
                CollectCPULoadAndRuntimeData(result0, result1, result2, result3, analysis_result);
            else if (strcmp(task->cuda, "gpu") == 0)
                CollectGPULoadAndRuntimeData(result0, result1, result2, result3, analysis_result);
        }
    }

    state->last_action = AgentAction::EXECUTION;
    return AgentAction::SCHEDULE;
}

// evaluation agent: evaluate the execution result
AgentAction EvaluationAgent::Execute(std::shared_ptr<AgentState> state) {
    Task* task = (Task*)list_nth(state->task_list, state->current_task_id);
    
    // 【关键修正 3】: 配合 Execution 的修改
    // 现在 outs 列表只包含当前 Task 的结果，索引从 0 开始
    int result_count = list_length(state->current_state.outs);
    int expected_count = task->input_end_index - task->input_start_index;

    if (result_count != expected_count) {
        elog(WARNING, "Evaluation warning: expected %d results, got %d", expected_count, result_count);
    }

    for (int i = 0; i < result_count; i++) {
        Args* ret = (Args*)list_nth(state->current_state.outs, i);
        // 计算对应的全局行号用于日志
        long global_row_index = task->input_start_index + i;
        memory_manager.out_cache_data[memory_manager.out_cache_size++] = ret->floating;
        // elog(INFO, "Evaluation Result on index %ld: %f", global_row_index, ret->floating);
    }

    // 任务完成后，可以在这里选择性释放 current_state.outs 里的 Args* 内存，但这通常由 PG 上下文自动处理
    
    state->last_action = AgentAction::EVALUATION;
    return AgentAction::SCHEDULE;
}

AgentAction ScheduleAgent::Execute(std::shared_ptr<AgentState> state) {
    // to be done
    try {
        if (state->last_action == AgentAction::START) {
            return AgentAction::PERCEPTION;
        } else if (state->last_action == AgentAction::PERCEPTION) {
            bool ready_for_task = (memory_manager.current_func_call % WINDOW_SIZE == WINDOW_SIZE - 1) ||
                              (memory_manager.is_last_call == 1);
            if (!ready_for_task) {
                return AgentAction::SUCCESS;
            }
            return AgentAction::ORCHESTRATION;
        } else if (state->last_action == AgentAction::ORCHESTRATION) {
            return AgentAction::OPTIMIZATION;
        } else if (state->last_action == AgentAction::OPTIMIZATION) {
            // to be done: 决策执行什么任务，若无任务可执行应当直接调到结束状态
            if (list_length(state->task_list) == 0) {
                return AgentAction::SUCCESS;
            }
            state->current_task_id = 0;
            return AgentAction::EXECUTION;
        } else if (state->last_action == AgentAction::EXECUTION) {
            return AgentAction::EVALUATION;
        } else if (state->last_action == AgentAction::EVALUATION) {
            state->task_list = list_delete_first(state->task_list);
            if (list_length(state->task_list) == 0) {
                return AgentAction::SUCCESS;
            } else {                
                state->current_task_id = 0;
                return AgentAction::EXECUTION;
            } 
        } else {
            return AgentAction::SUCCESS;
        }
    } catch (const std::exception& e) {
        elog(INFO, "ScheduleAgent error message:%s", e.what());
        return AgentAction::FAILURE;
    }
}