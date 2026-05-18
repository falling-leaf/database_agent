#pragma once
#ifndef _MODEL_AGENT_H_
#define _MODEL_AGENT_H_

/**
 * @file model_agent.h
 * @brief 模型代理系统的核心头文件，定义了内存管理、代理节点和任务管理等核心组件
 * 
 * 该文件包含了模型代理系统的核心架构定义，包括：
 * 1. MemoryManager：内存管理类，负责数据缓存和加载
 * 2. 各种代理节点类：负责不同阶段的任务处理
 * 3. 任务相关的数据结构：定义了任务类型、任务信息等
 * 4. 模型分析和资源监控相关的数据结构和函数
 */

#include "env.h"
#include "model_manager.h"
#include "model_selection.h"
#include "batch_interface.h"
#include "myfunc.h"
#include "vector.h"
#include "cpu_load_predictor.h"
#include "gpu_load_predictor.h"

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

/**
 * @brief 最大缓存大小
 */
#define MAX_CACHE_SIZE 100000

#include <chrono>
#include <unordered_map>

/**
 * @class MemoryManager
 * @brief 内存管理类，负责数据缓存、加载和管理
 * 
 * 该类提供了以下功能：
 * 1. 从数据库表中加载数据行
 * 2. 将元组转换为向量
 * 3. 管理输入和输出缓存
 * 4. 跟踪代理执行时间
 */
class MemoryManager {
public:
    /**
     * @brief 构造函数
     */
    MemoryManager() {}
    
    /**
     * @brief 析构函数
     */
    ~MemoryManager() {}
    
    /**
     * @brief 从指定表中加载一行数据
     * @param table_name 表名
     * @param row_index 行索引
     * @return 加载的数据行
     */
    static Args* LoadOneRow(const std::string& table_name, size_t row_index);
    
    /**
     * @brief 使用SPI从指定表中加载一行数据
     * @param tuple 输出参数，存储加载的元组
     * @param tupdesc 输出参数，存储元组描述符
     * @param table_name 表名
     * @param row_index 行索引
     */
    static void SPILoadOneRow(HeapTuple& tuple, TupleDesc& tupdesc, const std::string& table_name, size_t row_index);
    
    /**
     * @brief 将元组转换为向量
     * @param tuple 输入元组
     * @param tupdesc 元组描述符
     * @param start 起始列索引
     * @param dim 向量维度
     * @return 转换后的向量
     */
    static Args* Tuple2Vec(HeapTuple tuple, TupleDesc tupdesc, int start, int dim);
    
    /**
     * @brief 从指定表中加载文本数据（query和context）
     * @param table_name 表名
     * @param row_index 行索引
     * @param query 输出参数，存储查询文本
     * @param context 输出参数，存储上下文文本
     * @return 是否加载成功
     */
    static bool LoadTextData(const std::string& table_name, size_t row_index, std::string& query, std::string& context);

    /**
     * @brief 当前函数调用计数
     */
    int current_func_call{-1};
    
    /**
     * @brief 是否为最后一次调用
     * 0: 不是最后一次调用
     * 1: 是最后一次调用
     * 2: 未设置
     */
    int is_last_call{2};

    /**
     * @brief 输入缓存数据
     */
    float** ins_cache_data;
    
    /**
     * @brief 输入向量缓存
     */
    MVec** ins_cache;
    
    /**
     * @brief 输出缓存大小
     */
    int out_cache_size{0};
    
    /**
     * @brief 输出缓存数据
     */
    float* out_cache_data;
    
    /**
     * @brief 输出字符串缓存数据
     */
    char** out_cache_string_data;
    
    /**
     * @brief 输出索引
     */
    int output_index{0};
    
    /**
     * @brief 输出类型
     * 0: float8
     * 1: string
     */
    int output_type{0};
    
    /**
     * @brief 输入缓冲区
     */
    Args* ins_buffer{nullptr};

    /**
     * @brief 第二个输入缓存数据
     */
    float** ins2_cache_data;
    
    /**
     * @brief 第二个输入向量缓存
     */
    MVec** ins2_cache;

    /**
     * @brief 样本路径列表
     */
    std::vector<std::string> sample_path;
    
    /**
     * @brief 执行时间跟踪
     * 键: 代理名称
     * 值: 累计执行时间（微秒）
     */
    std::unordered_map<std::string, long long> execution_time_map;
    
    /**
     * @brief 执行次数跟踪
     * 键: 代理名称
     * 值: 执行次数
     */
    std::unordered_map<std::string, int> execution_count_map;
    
    /**
     * @brief 打印时间统计信息
     */
    void PrintTimingStats();
};

/**
 * @enum AgentAction
 * @brief 代理执行动作枚举
 */
enum class AgentAction {
    START,     ///< 开始执行
    SUCCESS,   ///< 执行成功
    FAILURE,   ///< 执行失败
    PERCEPTION, ///< 感知阶段
    ORCHESTRATION, ///< 编排阶段
    OPTIMIZATION, ///< 优化阶段
    EXECUTION, ///< 执行阶段
    EVALUATION, ///< 评估阶段
    SCHEDULE   ///< 调度阶段
};

/**
 * @enum TaskType
 * @brief 任务类型枚举
 */
enum class TaskType {
    IMAGE_CLASSIFICATION, ///< 图像分类任务
    SERIES, ///< 序列任务
    NLP, ///< 自然语言处理任务
    REASONING, ///< 推理任务
    REASONING_STEP2, ///< 推理第二步
    REASONING_STEP3, ///< 推理第三步
    STEP1, ///< 第一步任务
    STEP2, ///< 第二步任务
    STEP3 ///< 第三步任务
};

/**
 * @struct Task
 * @brief 任务结构
 * 
 * 定义了一个任务的详细信息，包括任务类型、模型、设备和数据范围等
 */
typedef struct Task {
    TaskType task_type; ///< 任务类型
    char* model; ///< 模型名称
    char* cuda; ///< 设备名称（cpu或gpu）
    char* table_name; ///< 表名

    int64_t input_start_index; ///< 输入起始索引
    int64_t input_end_index; ///< 输入结束索引
    int64_t output_start_index; ///< 输出起始索引
    int64_t output_end_index; ///< 输出结束索引
} Task;

/**
 * @struct TaskInfo
 * @brief 任务信息结构
 * 
 * 定义了任务的基本信息，用于任务初始化和管理
 */
typedef struct TaskInfo {
    TaskType task_type; ///< 任务类型
    char* table_name; ///< 表名
    char* model_name; ///< 模型名称
    char* cuda_name; ///< 设备名称
} TaskInfo;

/**
 * @struct VecAggState
 * @brief 向量聚合状态结构
 * 
 * 定义了向量聚合过程中的状态信息
 */
typedef struct VecAggState {
    MemoryContext ctx; ///< 内存上下文
    List* ins; ///< 输入列表
    List* outs; ///< 输出列表
    int batch_i; ///< 批处理索引
    int prcsd_batch_n; ///< 已处理批次数
    char* model; ///< 模型名称
    char* cuda; ///< 设备名称
    int nxt_csr; ///< 下一个CSR索引
    int64_t pre_time; ///< 预处理时间（毫秒）
    int64_t infer_time; ///< 推理时间（毫秒）
    int64_t post_time; ///< 后处理时间（毫秒）
} VecAggState;

/**
 * @struct AgentState
 * @brief 代理状态结构
 * 
 * 定义了代理执行过程中的状态信息
 */
typedef struct AgentState {
    FunctionCallInfo fcinfo; ///< 函数调用信息
    VecAggState current_state; ///< 当前状态
    AgentAction last_action; ///< 上一个动作
    int current_start_index; ///< 当前起始索引
    int current_end_index; ///< 当前结束索引
    int current_task_id; ///< 当前任务ID（task_list的下标）
    List* task_list; ///< 任务列表
    std::vector<TaskInfo> task_info; ///< 任务信息列表
} AgentState;

/**
 * @class BaseAgentNode
 * @brief 代理节点基类
 * 
 * 所有代理节点的基类，定义了基本接口
 */
class BaseAgentNode {
public:
    /**
     * @brief 虚析构函数
     */
    virtual ~BaseAgentNode() = default;

    /**
     * @brief 核心执行接口
     * @param state 代理状态
     * @return 执行动作
     */
    virtual AgentAction Execute(std::shared_ptr<AgentState> state) = 0;
    
    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    virtual std::string Name() const = 0;
    
    /**
     * @brief 带时间统计的执行包装器
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction ExecuteWithTiming(std::shared_ptr<AgentState> state);
};

/**
 * @class PerceptionAgent
 * @brief 感知代理
 * 
 * 负责将自然语言输入转换为嵌入向量
 */
class PerceptionAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行感知任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    
    /**
     * @brief 加载向量数据
     * @param input 输入向量
     */
    void LoadMVecData(MVec* input);

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "PerceptionAgent";
    }
};

/**
 * @class OrchestrationAgent
 * @brief 编排代理
 * 
 * 负责模型选择和资源管理
 */
class OrchestrationAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行编排任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;
    
    /**
     * @brief 注册SPI进程
     */
    void SPIRegisterProcess();
    
    /**
     * @brief 初始化任务
     * @param state 代理状态
     * @param window_start_index 窗口起始索引
     * @param window_end_index 窗口结束索引
     */
    void TaskInit(std::shared_ptr<AgentState> state, int window_start_index = -1, int window_end_index = -1);
    
    /**
     * @brief 选择模型
     * @param state 代理状态
     * @param task_type 任务类型
     * @return 模型名称
     */
    std::string SelectModel(std::shared_ptr<AgentState> state, TaskType task_type);
    
    /**
     * @brief 初始化模型
     * @param model_name 模型名称
     * @param from_select_model 是否来自模型选择
     * @return 是否初始化成功
     */
    bool InitModel(const char* model_name, bool from_select_model);

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "OrchestrationAgent";
    }
};

/**
 * @class OptimizationAgent
 * @brief 优化代理
 * 
 * 负责计划树优化和资源调度
 */
class OptimizationAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行优化任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "OptimizationAgent";
    }
};

/**
 * @class ExecutionAgent
 * @brief 执行代理
 * 
 * 负责执行计划
 */
class ExecutionAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "ExecutionAgent";
    }
};

/**
 * @class EvaluationAgent
 * @brief 评估代理
 * 
 * 负责评估执行结果
 */
class EvaluationAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行评估任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "EvaluationAgent";
    }
};

/**
 * @class ScheduleAgent
 * @brief 调度代理
 * 
 * 负责管理所有代理的执行顺序
 */
class ScheduleAgent : public BaseAgentNode {
public:
    /**
     * @brief 执行调度任务
     * @param state 代理状态
     * @return 执行动作
     */
    AgentAction Execute(std::shared_ptr<AgentState> state) override;

    /**
     * @brief 获取节点名称
     * @return 节点名称
     */
    std::string Name() const override {
        return "ScheduleAgent";
    }
};

/**
 * @struct ModelAnalysisResult
 * @brief 模型分析结果结构
 * 
 * 存储模型的特征信息
 */
struct ModelAnalysisResult {
    long long mac_count;      ///< 乘加操作次数
    long long param_count;    ///< 参数总数
    size_t param_size_bytes;  ///< 参数总大小（字节）
};

/**
 * @struct CpuLoadData
 * @brief CPU负载数据结构
 * 
 * 存储CPU负载信息
 */
struct CpuLoadData {
    double cpu_load; ///< CPU负载
    int cpu_cores; ///< CPU核心数
};

/**
 * @struct GpuLoadData
 * @brief GPU负载数据结构
 * 
 * 存储GPU负载信息
 */
struct GpuLoadData {
    double cuda_cores; ///< CUDA核心数
    double gpu_freq; ///< GPU频率
    double util; ///< GPU利用率
    double mem_used; ///< 已用显存
    double mem_total; ///< 总显存
};

/**
 * @struct GpuStatus
 * @brief GPU状态结构
 * 
 * 存储GPU状态信息
 */
struct GpuStatus {
    int id; ///< GPU ID
    double util; ///< GPU利用率（0-100）
    double mem_used; ///< 已用显存（MiB）
    double mem_total; ///< 总显存（MiB）
};

/**
 * @brief 分析模型特征
 * @param model_name 模型名称
 * @return 模型分析结果
 */
ModelAnalysisResult AnalyzeModelWithInference(const std::string& model_name);

/**
 * @brief 估计CPU FLOPS
 * @return CPU FLOPS
 */
double estimate_cpu_flops();

/**
 * @brief 获取GPU FLOPS
 * @return GPU FLOPS
 */
double get_gpu_flops();

/**
 * @brief 获取CPU内存带宽
 * @return CPU内存带宽
 */
double get_cpu_mem_bandwidth();

/**
 * @brief 获取GPU内存带宽
 * @return GPU内存带宽
 */
double get_gpu_mem_bandwidth();

/**
 * @brief 获取CPU负载因子
 * @return CPU负载因子
 */
double get_cpu_load_factor();

/**
 * @brief 执行命令并返回输出
 * @param cmd 命令
 * @return 命令输出
 */
std::string exec_command(const char* cmd);

/**
 * @brief 修剪字符串
 * @param s 输入字符串
 * @return 修剪后的字符串
 */
std::string trim(const std::string& s);

/**
 * @brief 获取所有GPU状态
 * @return GPU状态列表
 */
std::vector<GpuStatus> get_all_gpu_status();

/**
 * @brief 收集GPU负载数据
 * @return GPU负载数据
 */
GpuLoadData CollectGPULoadData();

/**
 * @brief 收集CPU负载数据
 * @return CPU负载数据
 */
CpuLoadData CollectCPULoadData();

/**
 * @brief 收集GPU负载和运行时数据
 * @param shape0 数据形状0
 * @param shape1 数据形状1
 * @param shape2 数据形状2
 * @param shape3 数据形状3
 * @param execution_runtime_us 执行运行时间（微秒）
 * @param analysis_result 模型分析结果
 * @param gpu_data GPU负载数据
 */
void CollectGPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, long long execution_runtime_us, ModelAnalysisResult analysis_result, GpuLoadData gpu_data);

/**
 * @brief 收集CPU负载和运行时数据
 * @param shape0 数据形状0
 * @param shape1 数据形状1
 * @param shape2 数据形状2
 * @param shape3 数据形状3
 * @param execution_runtime_us 执行运行时间（微秒）
 * @param analysis_result 模型分析结果
 * @param cpu_data CPU负载数据
 */
void CollectCPULoadAndRuntimeData(int shape0, int shape1, int shape2, int shape3, long long execution_runtime_us, ModelAnalysisResult analysis_result, CpuLoadData cpu_data);

/**
 * @brief 临时添加函数，用于处理推理任务的额外逻辑
 * @param state 代理状态
 */
void temp_addition_function(std::shared_ptr<AgentState> state);

#endif