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
    
    cpu_load_during = get_cpu_load()
    
    successful_tasks = [r for r in results if r["status"] == "success"]
    failed_tasks = [r for r in results if r["status"] == "failed"]
    
    success_rate = len(successful_tasks) / len(results) if len(results) > 0 else 0
    
    avg_latency = sum(r["latency"] for r in successful_tasks) / len(successful_tasks) if successful_tasks else 0
    min_latency = min((r["latency"] for r in successful_tasks), default=0)
    max_latency = max((r["latency"] for r in successful_tasks), default=0)
    
    print(f"Concurrency level {concurrency_level}: {len(successful_tasks)}/{len(results)} succeeded")
    print(f"  Avg latency: {avg_latency:.4f}s, Min: {min_latency:.4f}s, Max: {max_latency:.4f}s")
    print(f"  CPU Load (before): {cpu_load_before}, CPU Load (during): {cpu_load_during}")
    
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

    concurrency_levels = [15, 20, 25, 30]

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
        writer.writerow(['Query', 'Concurrency_Level', 'Avg_Latency', 'Min_Latency', 'Max_Latency', 'Success_Rate', 'CPU_Load_Before', 'CPU_Load_During'])
        for i, result in enumerate(results_1):
            writer.writerow(['db_agent_single', result['concurrency_level'], 
                           f"{result['avg_latency']:.4f}", f"{result['min_latency']:.4f}", 
                           f"{result['max_latency']:.4f}", f"{result['success_rate']:.2%}",
                           f"{result.get('cpu_load_before', 'N/A')}", f"{result.get('cpu_load_during', 'N/A')}"])
        for i, result in enumerate(results_2):
            writer.writerow(['predict_batch_float8', result['concurrency_level'], 
                           f"{result['avg_latency']:.4f}", f"{result['min_latency']:.4f}", 
                           f"{result['max_latency']:.4f}", f"{result['success_rate']:.2%}",
                           f"{result.get('cpu_load_before', 'N/A')}", f"{result.get('cpu_load_during', 'N/A')}"])
    
    print(f"CSV results saved to: {csv_filename}")
    
    return all_results


if __name__ == "__main__":
    test_db_agent_single_queries()
