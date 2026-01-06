import psycopg2
import time
from concurrent.futures import ProcessPoolExecutor
from morphingdb_test.config import db_config

# --- 保持你原有的 parse_timing_info 等工具函数不变 ---

def single_task_worker(task_id, row_count, query_times=10, symbol='cpu', sql_query=None):
    """
    单个进程执行的任务单元
    """
    try:
        # 每个进程必须建立自己的连接
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # 记录该任务的内部起始时间
        start_time = time.time()
        
        # 执行业务SQL
        # 注意：这里使用了你原本的逻辑
        cur.execute("select register_process();")
        
        # Use the provided SQL query
        sql = sql_query.format(row_count=row_count, symbol=symbol)
        
        for i in range(query_times):
            cur.execute(sql)
            cur.fetchall() # 确保数据读取完毕
        
        # cur.execute("select print_cost();")
        # timing_raw = cur.fetchall()[0][0]
        
        end_time = time.time()
        conn.close()
        
        return {
            "task_id": task_id,
            "status": "success",
            "latency": end_time - start_time
            # "internal_timing": timing_raw # 如果需要深入分析PG内部耗时
        }
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}

def run_throughput_test(concurrency_list, total_tasks_per_level, rows_per_query, sql_query, query_times=10):
    """
    吞吐量测试主函数
    :param concurrency_list: 并发数列表，例如 [1, 5, 10, 20, 50]
    :param total_tasks_per_level: 每个并发等级下总共要完成的任务数
    :param rows_per_query: 每次查询处理的数据行数
    :param sql_query: SQL query to execute (with placeholders for {row_count} and {symbol})
    :param query_times: 每次查询重复执行的次数
    """
    print(f"{'Concurrency':<12} | {'Total Tasks':<12} | {'Total Time(s)':<15} | {'TPS':<10} | {'Avg Latency(s)':<15}")
    print("-" * 75)

    results_summary = []

    for concurrency in concurrency_list:
        start_wall_time = time.time()
        
        # 使用进程池模拟并发
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            # 提交任务
            futures = [executor.submit(single_task_worker, i, rows_per_query, query_times, 'cpu', sql_query) for i in range(total_tasks_per_level)]
            
            # 等待所有任务完成并收集结果
            task_results = [f.result() for f in futures]
        
        end_wall_time = time.time()
        
        # 计算统计数据
        total_elapsed = end_wall_time - start_wall_time
        success_tasks = [r for r in task_results if r['status'] == 'success']
        failed_tasks = [r for r in task_results if r['status'] == 'failed']
        
        tps = len(success_tasks) / total_elapsed if total_elapsed > 0 else 0
        avg_latency = sum(r['latency'] for r in success_tasks) / len(success_tasks) if success_tasks else 0
        
        print(f"{concurrency:<12} | {len(success_tasks):<12} | {total_elapsed:<15.4f} | {tps:<10.2f} | {avg_latency:<15.4f}")
        
        if failed_tasks:
            print(f"  [!] Warning: {len(failed_tasks)} tasks failed at concurrency {concurrency}")

        results_summary.append({
            "concurrency": concurrency,
            "tps": tps,
            "total_time": total_elapsed,
            "failed": len(failed_tasks)
        })
    
    return results_summary

# Define SQL queries to test
SQL_QUERIES = [
    {
        "name": "slice_predict",
        "query": "select predict_batch_float8('slice', '{symbol}', data) over (rows between current row and 31 following) from slice_test limit {row_count};"
    },
    {
        "name": "db_agent_predict",
        "query": "select db_agent('predict', sub_table.data) over (rows between current row and 31 following) FROM (SELECT * FROM slice_test) AS sub_table limit {row_count};"
    },
    {
        "name": "googlenet_predict",
        "query": "select predict_batch_float8('googlenet_cifar10', 'gpu', image_vector) over (rows between current row and 31 following) from cifar_image_vector_table limit {row_count};"
    },
    {
        "name": "db_agent_image_classification",
        "query": "select db_agent('image_classification', sub_table.image_path) over (rows between current row and 31 following) FROM (SELECT * FROM cifar_image_table limit 100) AS sub_table;"
    }
]

def run_all_throughput_tests():
    CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64] 
    TOTAL_TASKS = 128
    ROWS_PER_QUERY = 1000

    for sql_info in SQL_QUERIES:
        print(f"\n{'='*60}")
        print(f"Testing SQL: {sql_info['name']}")
        print(f"Query: {sql_info['query']}")
        print(f"Starting Throughput Test: Rows per query = {ROWS_PER_QUERY}")
        print(f"{'='*60}")
        
        run_throughput_test(CONCURRENCY_LEVELS, TOTAL_TASKS, ROWS_PER_QUERY, sql_info['query'], query_times=10)

if __name__ == "__main__":
    run_all_throughput_tests()