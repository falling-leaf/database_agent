import psycopg2
import time
import os
from concurrent.futures import ProcessPoolExecutor
from morphingdb_test.config import db_config
import json
import csv
from datetime import datetime


def get_cpu_load():
    """
    获取系统当前的平均负载 (CPU average load)
    读取 /proc/loadavg 文件获取1分钟平均负载
    """
    try:
        with open('/proc/loadavg', 'r') as f:
            content = f.read().strip()
            parts = content.split()
            cpu_load = float(parts[0])
            return cpu_load
    except Exception as e:
        print(f"Warning: Could not read CPU load: {e}")
        return None


def get_gpu_info():
    """
    获取 GPU 利用率和显存使用情况
    使用 nvidia-smi 命令获取 GPU 信息
    """
    try:
        import subprocess
        import re
        
        # 运行 nvidia-smi 命令
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                if len(parts) == 3:
                    gpu_util = int(parts[0].strip())
                    mem_used = int(parts[1].strip())
                    mem_total = int(parts[2].strip())
                    mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                    gpu_info.append({
                        'utilization': gpu_util,
                        'memory_used': mem_used,
                        'memory_total': mem_total,
                        'memory_utilization': mem_util
                    })
        
        return gpu_info
    except Exception as e:
        print(f"Warning: Could not read GPU info: {e}")
        return None


def run_single_task_worker(task_id, row_count, query_times, sql_query):
    """
    单个进程执行的任务单元
    """
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        start_time = time.time()
        
        cur.execute("select register_process();")
        
        sql = sql_query
        return_rows = 0
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall()
            return_rows += len(result)
            conn.commit()
        
        end_time = time.time()
        conn.close()
        
        return {
            "task_id": task_id,
            "status": "success",
            "latency": end_time - start_time,
            "return_rows": return_rows
        }
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}


def test_concurrency(concurrency_level, row_count, query_times, sql_query):
    """
    Test a specific concurrency level
    """
    print(f"Testing concurrency level: {concurrency_level}")
    
    cpu_load_before = get_cpu_load()
    gpu_info_before = get_gpu_info()
    
    # 用于存储 GPU 测量数据
    import threading
    import time
    gpu_measurements = []
    stop_measuring = False
    
    # 定期获取 GPU 信息的函数
    def measure_gpu(): 
        while not stop_measuring:
            gpu_info = get_gpu_info()
            if gpu_info:
                gpu_measurements.append(gpu_info)
            time.sleep(0.5)  # 每0.5秒测量一次，增加测量频率
    
    # 启动 GPU 测量线程
    gpu_thread = threading.Thread(target=measure_gpu)
    gpu_thread.daemon = True
    gpu_thread.start()
    
    # 给测量线程一点时间初始化
    time.sleep(0.2)
    
    with ProcessPoolExecutor(max_workers=concurrency_level) as executor:
        futures = []
        for i in range(concurrency_level):
            future = executor.submit(run_single_task_worker, i, row_count, query_times, sql_query)
            futures.append(future)
        
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=6000000)
                results.append(result)
            except Exception as e:
                print(f"Timeout or error in task {i}: {str(e)}")
                results.append({"task_id": i, "status": "failed", "error": f"Timeout or error: {str(e)}"})
    
    # 停止 GPU 测量
    stop_measuring = True
    gpu_thread.join(timeout=2)  # 等待测量线程结束
    
    cpu_load_during = get_cpu_load()
    
    # 计算 GPU 信息的最大值
    gpu_info_during = None
    if gpu_measurements:
        # 初始化最大值计算
        max_gpu_info = []
        num_gpus = len(gpu_measurements[0])
        
        # 对每个 GPU 计算最大值
        for gpu_idx in range(num_gpus):
            max_util = 0
            max_mem_used = 0
            max_mem_total = 0
            
            for measurement in gpu_measurements:
                if len(measurement) > gpu_idx:
                    max_util = max(max_util, measurement[gpu_idx]['utilization'])
                    max_mem_used = max(max_mem_used, measurement[gpu_idx]['memory_used'])
                    max_mem_total = max(max_mem_total, measurement[gpu_idx]['memory_total'])
            
            max_mem_util = (max_mem_used / max_mem_total) * 100 if max_mem_total > 0 else 0
            
            max_gpu_info.append({
                'utilization': max_util,
                'memory_used': max_mem_used,
                'memory_total': max_mem_total,
                'memory_utilization': max_mem_util
            })
        
        gpu_info_during = max_gpu_info
        print(f"Collected {len(gpu_measurements)} GPU measurements")
    
    successful_tasks = [r for r in results if r["status"] == "success"]
    failed_tasks = [r for r in results if r["status"] == "failed"]
    
    success_rate = len(successful_tasks) / len(results) if len(results) > 0 else 0
    
    avg_latency = sum(r["latency"] for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
    min_latency = min((r["latency"] for r in successful_tasks), default=0)
    max_latency = max((r["latency"] for r in successful_tasks), default=0)
    
    print(f"Concurrency level {concurrency_level}: {len(successful_tasks)}/{len(results)} succeeded")
    print(f"  Avg latency: {avg_latency:.4f}s, Min: {min_latency:.4f}s, Max: {max_latency:.4f}s")
    print(f"  CPU Load (before): {cpu_load_before}, CPU Load (during): {cpu_load_during}")
    
    # 打印 GPU 信息
    if gpu_info_before:
        print("  GPU Info (before):")
        for i, gpu in enumerate(gpu_info_before):
            print(f"    GPU {i}: Utilization: {gpu['utilization']}%, Memory: {gpu['memory_used']}MB/{gpu['memory_total']}MB ({gpu['memory_utilization']:.1f}%)")
    if gpu_info_during:
        print("  GPU Info (during - max):")
        for i, gpu in enumerate(gpu_info_during):
            print(f"    GPU {i}: Utilization: {gpu['utilization']:.1f}%, Memory: {gpu['memory_used']:.1f}MB/{gpu['memory_total']:.1f}MB ({gpu['memory_utilization']:.1f}%)")
    
    return {
        "concurrency_level": concurrency_level,
        "total_tasks": len(results),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "cpu_load_before": cpu_load_before,
        "cpu_load_during": cpu_load_during,
        "gpu_info_before": gpu_info_before,
        "gpu_info_during": gpu_info_during,
        "results": results,
        "row_count_per_task": row_count,
        "query_times_per_task": query_times
    }


def test_db_agent_single_queries():
    """
    Test db_agent_single queries under different concurrency levels
    """
    row_count = 1000
    query_times = 1
    
    # sql_query_1 = "select unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test limit 1000) AS sub_table;"
    # sql_query_2 = "select predict_batch_float8('slice', 'cpu', data) over (rows between current row and 31 following) from slice_test limit 1000;"
    
    # concurrency_levels = [1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400]

    sql_query_1 = "select unnest(db_agent_single('image_classification', sub_table.image_vector)) AS score FROM (SELECT * FROM stanford_dogs_image_vector_table limit 3200) AS sub_table;"
    sql_query_2 = "select predict_batch_float8('alexnet_stanford_dogs', 'gpu', image_vector) over (rows between current row and 31 following) from stanford_dogs_image_vector_table limit 3200;"

    # concurrency_levels = [1, 5, 10, 15]
    concurrency_levels = [1, 5, 10]

    results_1 = []
    results_2 = []
    
    print("="*80)
    print("Testing SQL Query 1: db_agent_single")
    print(f"Query: {sql_query_1}")
    print("="*80)
    
    for level in concurrency_levels:
        print(f"\n--- Testing with {level} concurrent tasks ---")
        result = test_concurrency(level, row_count, query_times, sql_query_1)
        results_1.append(result)
        
        if level < concurrency_levels[-1]:
            print(f"\n  Cooling down for 20 seconds...")
            time.sleep(20)
    
    print("\n" + "="*80)
    print("Testing SQL Query 2: predict_batch_float8")
    print(f"Query: {sql_query_2}")
    print("="*80)
    
    for level in concurrency_levels:
        print(f"\n--- Testing with {level} concurrent tasks ---")
        result = test_concurrency(level, row_count, query_times, sql_query_2)
        results_2.append(result)
        
        if level < concurrency_levels[-1]:
            print(f"\n  Cooling down for 20 seconds...")
            time.sleep(20)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "row_count": row_count,
        "query_times": query_times,
        "concurrency_levels": concurrency_levels,
        "query_1": {
            "sql": sql_query_1,
            "results": results_1
        },
        "query_2": {
            "sql": sql_query_2,
            "results": results_2
        }
    }
    
    json_filename = f"db_agent_single_comparison_results_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_filename}")
    
    csv_filename = json_filename.replace('.json', '.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Query', 'Concurrency_Level', 'Avg_Latency', 'Min_Latency', 'Max_Latency', 'Success_Rate', 
                       'CPU_Load_Before', 'CPU_Load_During', 
                       'GPU_Util_Before', 'GPU_Util_During (Max)', 'GPU_Mem_Used_Before', 'GPU_Mem_Used_During (Max)'])
        
        for i, result in enumerate(results_1):
            # 提取 GPU 信息
            gpu_util_before = 'N/A'
            gpu_util_during = 'N/A'
            gpu_mem_before = 'N/A'
            gpu_mem_during = 'N/A'
            
            if result.get('gpu_info_before'):
                gpu_util_before = result['gpu_info_before'][0]['utilization'] if len(result['gpu_info_before']) > 0 else 'N/A'
                gpu_mem_before = result['gpu_info_before'][0]['memory_used'] if len(result['gpu_info_before']) > 0 else 'N/A'
            if result.get('gpu_info_during'):
                gpu_util_during = result['gpu_info_during'][0]['utilization'] if len(result['gpu_info_during']) > 0 else 'N/A'
                gpu_mem_during = result['gpu_info_during'][0]['memory_used'] if len(result['gpu_info_during']) > 0 else 'N/A'
            
            # 确保数值格式正确
            if gpu_util_before != 'N/A':
                gpu_util_before = f"{gpu_util_before:.1f}"
            if gpu_util_during != 'N/A':
                gpu_util_during = f"{gpu_util_during:.1f}"
            if gpu_mem_before != 'N/A':
                gpu_mem_before = f"{gpu_mem_before:.1f}"
            if gpu_mem_during != 'N/A':
                gpu_mem_during = f"{gpu_mem_during:.1f}"
            
            writer.writerow(['db_agent_single', result['concurrency_level'], 
                           f"{result['avg_latency']:.4f}", f"{result['min_latency']:.4f}", 
                           f"{result['max_latency']:.4f}", f"{result['success_rate']:.2%}",
                           f"{result.get('cpu_load_before', 'N/A')}", f"{result.get('cpu_load_during', 'N/A')}",
                           gpu_util_before, gpu_util_during, gpu_mem_before, gpu_mem_during])
        
        for i, result in enumerate(results_2):
            # 提取 GPU 信息
            gpu_util_before = 'N/A'
            gpu_util_during = 'N/A'
            gpu_mem_before = 'N/A'
            gpu_mem_during = 'N/A'
            
            if result.get('gpu_info_before'):
                gpu_util_before = result['gpu_info_before'][0]['utilization'] if len(result['gpu_info_before']) > 0 else 'N/A'
                gpu_mem_before = result['gpu_info_before'][0]['memory_used'] if len(result['gpu_info_before']) > 0 else 'N/A'
            if result.get('gpu_info_during'):
                gpu_util_during = result['gpu_info_during'][0]['utilization'] if len(result['gpu_info_during']) > 0 else 'N/A'
                gpu_mem_during = result['gpu_info_during'][0]['memory_used'] if len(result['gpu_info_during']) > 0 else 'N/A'
            
            # 确保数值格式正确
            if gpu_util_before != 'N/A':
                gpu_util_before = f"{gpu_util_before:.1f}"
            if gpu_util_during != 'N/A':
                gpu_util_during = f"{gpu_util_during:.1f}"
            if gpu_mem_before != 'N/A':
                gpu_mem_before = f"{gpu_mem_before:.1f}"
            if gpu_mem_during != 'N/A':
                gpu_mem_during = f"{gpu_mem_during:.1f}"
            
            writer.writerow(['predict_batch_float8', result['concurrency_level'], 
                           f"{result['avg_latency']:.4f}", f"{result['min_latency']:.4f}", 
                           f"{result['max_latency']:.4f}", f"{result['success_rate']:.2%}",
                           f"{result.get('cpu_load_before', 'N/A')}", f"{result.get('cpu_load_during', 'N/A')}",
                           gpu_util_before, gpu_util_during, gpu_mem_before, gpu_mem_during])
    
    print(f"CSV results saved to: {csv_filename}")
    
    return all_results


if __name__ == "__main__":
    test_db_agent_single_queries()
