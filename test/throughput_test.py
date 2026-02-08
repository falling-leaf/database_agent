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
        # The sql_query should already be formatted with all necessary parameters
        sql = sql_query
        executed_rows = 0
        
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall() # 确保数据读取完毕
            executed_rows += len(result)  # 记录本次查询返回的行数
        expected_rows = row_count * query_times
        assert executed_rows == expected_rows, f"Expected {expected_rows} rows, but got {executed_rows}"
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
    :param sql_query: SQL query to execute (pre-formatted with all parameters)
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
            # Pass the pre-formatted SQL query without additional formatting
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

# Original SQL queries (predict_batch_float8) - 9 test cases across 3 types
ORIGINAL_SQL_QUERIES = [
    # Series tests (3) - use CPU
    {
        "name": "slice_predict",
        "table": "slice_test",
        "model": "slice",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "cpu"  # series tests use cpu
    },
    {
        "name": "swarm_predict", 
        "table": "swarm_test",
        "model": "swarm",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "cpu"  # series tests use cpu
    },
    {
        "name": "year_predict_test",
        "table": "year_predict_test", 
        "model": "year_predict",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "cpu"  # series tests use cpu
    },
    # NLP tests (3) - use GPU
    {
        "name": "imdb_vector_predict",
        "table": "imdb_vector_test",
        "model": "sst2_vec",
        "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # nlp tests use gpu
    },
    {
        "name": "financial_phrasebank_predict",
        "table": "financial_phrasebank_vector_test",
        "model": "finance",
        "column": "comment_vec", 
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # nlp tests use gpu
    },
    {
        "name": "nlp_vector_predict",
        "table": "nlp_vector_test",
        "model": "sst2_vec",  # Using sst2_vec as placeholder since exact model isn't clear
        "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # nlp tests use gpu
    },
    # Image tests (3) - use GPU
    {
        "name": "cifar_image_predict",
        "table": "cifar_image_vector_table",
        "model": "googlenet_cifar10",
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # image tests use gpu
    },
    {
        "name": "stanford_dogs_image_predict",
        "table": "stanford_dogs_image_vector_table", 
        "model": "alexnet_stanford_dogs",
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # image tests use gpu
    },
    {
        "name": "imagenet_image_predict",
        "table": "imagenet_image_vector_table",
        "model": "defect_vec", 
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};",
        "device": "gpu"  # image tests use gpu
    }
]

# New SQL queries based on the commented examples (db_agent_single) - 9 test cases across 3 types
NEW_SQL_QUERIES = [
    # Series tests using db_agent_single (3)
    {
        "name": "slice_db_agent",
        "table": "slice_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "cpu"  # series tests use cpu
    },
    {
        "name": "swarm_db_agent", 
        "table": "swarm_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "cpu"  # series tests use cpu
    },
    {
        "name": "year_predict_db_agent",
        "table": "year_predict_test", 
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "cpu"  # series tests use cpu
    },
    # NLP tests using db_agent_single (3)
    {
        "name": "imdb_db_agent",
        "table": "imdb_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # nlp tests use gpu
    },
    {
        "name": "financial_phrasebank_db_agent",
        "table": "financial_phrasebank_vector_test",
        "func_type": "nlp",
        "column": "comment_vec", 
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # nlp tests use gpu
    },
    {
        "name": "nlp_db_agent",
        "table": "nlp_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # nlp tests use gpu
    },
    # Image tests using db_agent_single (3)
    {
        "name": "cifar_db_agent",
        "table": "cifar_image_vector_table",
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # image tests use gpu
    },
    {
        "name": "stanford_dogs_db_agent",
        "table": "stanford_dogs_image_vector_table", 
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # image tests use gpu
    },
    {
        "name": "imagenet_db_agent",
        "table": "imagenet_image_vector_table",
        "func_type": "image_classification", 
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;",
        "device": "gpu"  # image tests use gpu
    }
]

def run_all_throughput_tests():
    # CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]
    # CONCURRENCY_LEVELS = [1, 4, 8]
    CONCURRENCY_LEVELS = [32, 64]
    TOTAL_TASKS = 128
    ROWS_PER_QUERY = 1000

    # Test original queries (predict_batch_float8)
    print("="*80)
    print("TESTING ORIGINAL QUERIES (predict_batch_float8)")
    print("="*80)
    
    for sql_info in ORIGINAL_SQL_QUERIES:
        print(f"\n{'='*80}")
        print(f"Testing SQL: {sql_info['name']}")
        print(f"Table: {sql_info['table']}")
        print(f"Model: {sql_info['model'] if 'model' in sql_info else sql_info.get('func_type', 'N/A')}")
        print(f"Device: {sql_info['device']}")
        print(f"Query: {sql_info['query']}")
        print(f"Starting Throughput Test: Rows per query = {ROWS_PER_QUERY}")
        print(f"{'='*80}")
        
        # Format the query with actual values
        if 'model' in sql_info:
            # For original queries with model
            formatted_query = sql_info['query'].format(
                model=sql_info['model'],
                symbol=sql_info['device'],
                column=sql_info['column'],
                table=sql_info['table'],
                row_count=ROWS_PER_QUERY
            )
        else:
            # For NEW queries with func_type (though this shouldn't happen in ORIGINAL_SQL_QUERIES)
            formatted_query = sql_info['query'].format(
                func_type=sql_info['func_type'],
                column=sql_info['column'],
                table=sql_info['table']
            ).format(row_count=ROWS_PER_QUERY)  # Handle the double brace for row_count
        
        run_throughput_test(CONCURRENCY_LEVELS, TOTAL_TASKS, ROWS_PER_QUERY, formatted_query, query_times=1)

    # Test new queries (db_agent_single)
    print("\n" + "="*80)
    print("TESTING NEW QUERIES (db_agent_single)")
    print("="*80)
    
    for sql_info in NEW_SQL_QUERIES:
        print(f"\n{'='*80}")
        print(f"Testing SQL: {sql_info['name']}")
        print(f"Table: {sql_info['table']}")
        print(f"Function Type: {sql_info['func_type']}")
        print(f"Device: {sql_info['device']}")
        print(f"Query: {sql_info['query']}")
        print(f"Starting Throughput Test: Rows per query = {ROWS_PER_QUERY}")
        print(f"{'='*80}")
        
        # Format the query with actual values
        # For NEW queries with func_type
        formatted_query_template = sql_info['query'].format(
            func_type=sql_info['func_type'],
            column=sql_info['column'],
            table=sql_info['table']
        )
        # Replace the row_count placeholder - note: the template uses {{row_count}} which becomes {row_count} after first format
        formatted_query = formatted_query_template.format(row_count=ROWS_PER_QUERY)
        
        run_throughput_test(CONCURRENCY_LEVELS, TOTAL_TASKS, ROWS_PER_QUERY, formatted_query, query_times=1)

if __name__ == "__main__":
    run_all_throughput_tests()