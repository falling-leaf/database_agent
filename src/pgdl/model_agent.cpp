#include "model_agent.h"
#include "model_manager.h"
#include "model_utils.h"
#include "vector.h"
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
#include "utils/memutils.h"
#include "miscadmin.h"

extern void infer_batch_internal(VecAggState *state, bool ret_float8);
}
extern void register_callback();
extern ModelManager model_manager;
extern MemoryManager memory_manager;

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


AgentAction initialize_state(std::shared_ptr<AgentState> state) {
    // Initialize the agent state
    state->last_action = AgentAction::START;
    // to be done
    return AgentAction::SCHEDULE;
}

void PerceptionAgent::LoadMVecData(MVec* current_data) {
    if (IS_MVEC_REF(current_data)) {
        ereport(ERROR, (errmsg("Input vector is a Reference (RowId: %ld), but api_agent requires materialized data. "
                               "Please use de-referencing functions or ensure input is a concrete vector.", 
                               (long)GET_MVEC_ROWID(current_data))));
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

// perception agent: NL =====> embedding vector
AgentAction PerceptionAgent::Execute(std::shared_ptr<AgentState> state) {
    FunctionCallInfo fcinfo = state->fcinfo;
    // elog(INFO, "the number of param: %d", PG_NARGS());
    for (int i = 0; i < PG_NARGS(); i++) {
        if (PG_ARGISNULL(i)) {
            elog(INFO, "param %d is null", i);
            throw std::runtime_error("NULL param accepted.");
        }
    }
    TaskInfo task_info;
    int load_index = 0;
    char* task_type = (char*)PG_GETARG_CSTRING(load_index++);
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
    if (memory_manager.total_func_call == 0) {
        memory_manager.total_func_call = PG_GETARG_INT32(load_index++);
        if (memory_manager.ins_cache == NULL)
        {
            elog(INFO, "reset the space");
            MemoryContext old_context = MemoryContextSwitchTo(TopMemoryContext);
            memory_manager.ins_cache_data = (float**)palloc0(sizeof(float*) * memory_manager.total_func_call);
            memory_manager.ins_cache = (MVec**)palloc0(sizeof(MVec*) * memory_manager.total_func_call);
            MemoryContextSwitchTo(old_context);
            
            elog(NOTICE, "Global memory allocated in TopMemoryContext.");
        }
        memory_manager.current_func_call = 0;
    } else {
        load_index++;
        memory_manager.current_func_call++;
    }
    if (memory_manager.current_func_call >= memory_manager.total_func_call) {
        throw std::runtime_error("variable fatal error in call times, restart the system.");
    }
    // elog(INFO, "test for total: %d, current: %d", memory_manager.total_func_call, memory_manager.current_func_call);
    // TODO：下面代码应当只会被执行一次
    if (memory_manager.current_func_call == 0) {
        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
        task_info.table_name = (char*)PG_GETARG_CSTRING(load_index++);
        // task_info.limit_length = atoi((char*)PG_GETARG_CSTRING(load_index++));
        if (task_info.task_type == TaskType::IMAGE_CLASSIFICATION) {
            task_info.select_table_name = (char*)PG_GETARG_CSTRING(load_index++);
            task_info.col_name = (char*)PG_GETARG_CSTRING(load_index++);
        }
        state->task_info.emplace_back(task_info);
        MemoryContextSwitchTo(old_ctx);
    } else {
        if (task_info.task_type == TaskType::IMAGE_CLASSIFICATION)
            load_index += 3;
        else load_index += 1;
    }
    // 下面代码每次调用均会执行
    MVec* current_data = (MVec*)PG_GETARG_MVEC_P(load_index++);
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
    if (memory_manager.current_func_call == 0)
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
    // state->current_start_index = 0;
    // state->current_end_index = 0;
    for (auto& task_unit : state->task_info) {
        // 先默认窗口为32
        int window_size = 32;
        std::string selected_model;
        if (memory_manager.current_func_call == 0) {
            elog(INFO, "start setting model and cuda");
            switch (task_unit.task_type) {
                case TaskType::IMAGE_CLASSIFICATION:
                    {
                        selected_model = SelectModel(state, task_unit.select_table_name, task_unit.col_name);
                        // 关键修复：使用 pstrdup 复制字符串到 PostgreSQL 内存上下文
                        MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                        task_unit.model_name = pstrdup(selected_model.c_str());
                        task_unit.cuda_name = pstrdup("gpu");   
                        MemoryContextSwitchTo(old_ctx);
                        if (InitModel(selected_model.c_str())) {
                            elog(INFO, "InitModel success");
                        }
                        else {
                            elog(ERROR, "InitModel failed");
                        }
                    }
                    break;
                case TaskType::PREDICT:
                    {
                        if (strcmp(task_unit.table_name, "slice_test") == 0) {
                            window_size = 32;
                            MemoryContext old_ctx = MemoryContextSwitchTo(TopMemoryContext);
                            task_unit.model_name = pstrdup("slice");
                            task_unit.cuda_name = pstrdup("cpu");
                            MemoryContextSwitchTo(old_ctx);
                        }
                    }
                    break;
                default:
                    elog(ERROR, "unknown task type %d", task_unit.task_type);
                    break;
            }
        }
        bool ready_for_task = (memory_manager.current_func_call % window_size == window_size - 1) ||
                              (memory_manager.current_func_call + 1 == memory_manager.total_func_call);
        if (!ready_for_task)
            continue;

        int64_t window_start = (memory_manager.current_func_call / window_size) * window_size;
        int64_t window_end = window_start + window_size;
        if (window_end > memory_manager.total_func_call) {
            window_end = memory_manager.total_func_call;
        }
        Task* task = (Task*)palloc0(sizeof(Task));
        if (task == NULL) {
            elog(ERROR, "Failed to allocate memory for Task");
            throw std::bad_alloc();
        }
        // 关键修复：为每个 task 复制字符串，而不是共享同一个指针
        task->model = pstrdup(task_unit.model_name);
        task->cuda = pstrdup(task_unit.cuda_name);
        task->table_name = task_unit.table_name;
        task->input_start_index = window_start;
        task->input_end_index = window_end; // 包含start，不包含end
        task->output_start_index = 0;
        task->output_end_index = task->input_end_index - task->input_start_index;
        state->task_list = lappend(state->task_list, task);
    }
    // elog(INFO, "task_list length: %d", list_length(state->task_list));
    for (int i = 0; i < list_length(state->task_list); i++) {
        Task* task = (Task*)list_nth(state->task_list, i);
        elog(INFO, "task %d: model: %s, cuda: %s, table_name: %s, input_start_index: %ld, input_end_index: %ld, output_start_index: %ld, output_end_index: %ld", i, task->model, task->cuda, task->table_name, task->input_start_index, task->input_end_index, task->output_start_index, task->output_end_index);
    }
}

std::string OrchestrationAgent::SelectModel(std::shared_ptr<AgentState> state, const std::string& select_table_name, const std::string& col_name) {
    std::string select_model_path = "/home/why/pgdl/model/models/select_model/ViT-L-14_visual_traced.pt";
    std::string regression_model_path = "/home/why/pgdl/model/models/select_model/regression_model.onnx";
    ModelSelection image_classification(select_model_path, regression_model_path);
    int sample_size = 10;

    std::string finetune_ds;
    size_t pos = select_table_name.find('_');
    if (pos != std::string::npos) {
        finetune_ds = select_table_name.substr(0, pos);
    } else {
        elog(INFO, "Here");
        finetune_ds = select_table_name;
    }
    if (finetune_ds == "cifar") {
        finetune_ds += "10";
    }
    std::string temp_result = image_classification.SelectModel(select_table_name, col_name, sample_size, finetune_ds);
    pos = temp_result.find('%');
    if (pos == std::string::npos) {
        // 没找到 %, 根据需求处理
        throw std::runtime_error("SelectModel: invalid result string");
    }
    if (select_table_name.size() < 12) {
        throw std::runtime_error("SelectModel: invalid select_table_name");
    }
    std::string arch = temp_result.substr(0, pos);
    std::string pretrain_ds = temp_result.substr(pos + 1);
    std::string result_str = arch + "_" + arch + "_" + finetune_ds + "_" + "from" + "_" + pretrain_ds;
    return result_str;
}

bool OrchestrationAgent::InitModel(const char* model_name) {
    std::string full_path = "/home/why/models_all/";
    full_path += model_name;
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
        // if(parameter_list == NULL){
        //     ereport(ERROR, (errmsg("model_parameter_extraction error!")));
        // }

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

    // 设置模型参数
    state->current_state.model = task->model;
    state->current_state.cuda = task->cuda;

    // 【关键修正 2】: 抛弃“滑动窗口”逻辑，每次执行 Task 前彻底重置 Input/Output 列表
    // 之前的逻辑导致 outs 列表无限增长，导致第二次 infer 时崩溃
    state->current_state.ins = NIL;
    state->current_state.outs = NIL;

    // 重新构建本批次的输入列表
    // 直接从 ins_cache 中根据 task 的索引范围抓取数据
    for (int64_t i = task->input_start_index; i < task->input_end_index; i++) {
        // 分配一个新的 Args 容器来承载指针 (注意：ins_cache 本身不需要拷贝)
        Args* vec = (Args*)palloc0(sizeof(Args));
        
        // 防御性检查
        if (memory_manager.ins_cache[i] == NULL) {
             ereport(ERROR, (errmsg("ins_cache is null at index %ld", i)));
        }
        
        vec->ptr = memory_manager.ins_cache[i];
        
        // 将其加入输入列表
        state->current_state.ins = lappend(state->current_state.ins, vec);
    }

    elog(INFO, "Constructed batch size: %d. Starting inference...", list_length(state->current_state.ins));

    // 执行推理
    // infer_batch_internal 会将结果填入 state->current_state.outs
    infer_batch_internal(&state->current_state, true);

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
        elog(INFO, "Evaluation Result on index %ld: %f", global_row_index, ret->floating);
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