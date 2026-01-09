import psycopg2
import time
from concurrent.futures import ProcessPoolExecutor
from morphingdb_test.config import db_config

# --- 保持你原有的 parse_timing_info 等工具函数不变 ---

def run_single_task_worker(task_id, row_count, query_times=10, symbol='cpu', sql_query=None):
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
        return_rows = 0
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall() # 确保数据读取完毕
            return_rows += len(result)
        expected_rows = row_count * query_times
        print(f"return_rows: {return_rows}, expected_rows: {expected_rows}")
        
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

# Define SQL queries to test
SQL_QUERIES = [
    {
        "name": "slice_predict",
        "query": "select predict_batch_float8('slice', '{symbol}', data) over (rows between current row and 31 following) from slice_test limit {row_count};"
    },
    {
        "name": "db_agent_predict",
        "query": "select unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test limit 1000) AS sub_table limit {row_count};"
    }
]

IMAGE_SQL_QUERIES = [
    {
        "name": "googlenet_predict",
        "query": "select predict_batch_float8('googlenet_cifar10', 'gpu', image_vector) over (rows between current row and 31 following) from cifar_image_vector_table limit {row_count};"
    },
    {
        "name": "db_agent_image_classification",
        "query": "select db_agent_batch('image_classification', sub_table.image_vector) over (rows between current row and 31 following) FROM (SELECT * FROM cifar_image_vector_table) AS sub_table limit {row_count};"
    }
]

def run_single_slice_test():
    for sql_info in IMAGE_SQL_QUERIES:
        print(f"\n{'='*60}")
        print(f"Testing SQL: {sql_info['name']}")
        print(f"Query: {sql_info['query']}")
        print(f"Starting Single Slice Test: Rows per query = 1000")
        print(f"{'='*60}")
        
        result = run_single_task_worker(0, 1000, 10, 'cpu', sql_info['query'])
        print(result)

if __name__ == "__main__":
    run_single_slice_test()
