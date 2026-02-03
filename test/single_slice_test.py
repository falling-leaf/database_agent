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
        # start_time = time.time()
        
        # 执行业务SQL
        # 注意：这里使用了你原本的逻辑
        cur.execute("select register_process();")
        start_time = time.time()
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

# select unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM slice_test limit 70) AS sub_table;
# select unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM swarm_test limit 70) AS sub_table;
# select unnest(db_agent_single('series', sub_table.data)) AS score FROM (SELECT * FROM year_predict_test limit 70) AS sub_table;

# select unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM imdb_vector_test limit 70) AS sub_table;
# select unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM financial_phrasebank_vector_test limit 70) AS sub_table;
# select unnest(db_agent_single('nlp', sub_table.comment_vec)) AS score FROM (SELECT * FROM nlp_vector_test limit 70) AS sub_table;

# select unnest(db_agent_single('image_classification', sub_table.image_path)) AS score FROM (SELECT * FROM cifar_image_table limit 100) AS sub_table;
# select unnest(db_agent_single('image_classification', sub_table.image_path)) AS score FROM (SELECT * FROM stanford_dogs_image_table limit 100) AS sub_table;
# select unnest(db_agent_single('image_classification', sub_table.image_path)) AS score FROM (SELECT * FROM imagenet_image_table limit 100) AS sub_table;

# Define SQL queries to test - 9 test cases across 3 types
SQL_QUERIES = [
    # Series tests (3)
    {
        "name": "slice_predict",
        "table": "slice_test",
        "model": "slice",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "swarm_predict", 
        "table": "swarm_test",
        "model": "swarm",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "year_predict_test",
        "table": "year_predict_test", 
        "model": "year_predict",
        "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    # NLP tests (3)
    {
        "name": "imdb_vector_predict",
        "table": "imdb_vector_test",
        "model": "sst2_vec",
        "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "financial_phrasebank_predict",
        "table": "financial_phrasebank_vector_test",
        "model": "finance",
        "column": "comment_vec", 
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "nlp_vector_predict",
        "table": "nlp_vector_test",
        "model": "sst2_vec",  # Using sst2_vec as placeholder since exact model isn't clear
        "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    # Image tests (3)
    {
        "name": "cifar_image_predict",
        "table": "cifar_image_vector_table",
        "model": "googlenet_cifar10",
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "stanford_dogs_image_predict",
        "table": "stanford_dogs_image_vector_table", 
        "model": "alexnet_stanford_dogs",
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "imagenet_image_predict",
        "table": "imagenet_image_vector_table",
        "model": "defect_vec", 
        "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    }
]

# New SQL queries based on the commented examples
NEW_SQL_QUERIES = [
    # Series tests using db_agent_single (3)
    {
        "name": "slice_db_agent",
        "table": "slice_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    # {
    #     "name": "swarm_db_agent", 
    #     "table": "swarm_test",
    #     "func_type": "series",
    #     "column": "data",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # {
    #     "name": "year_predict_db_agent",
    #     "table": "year_predict_test", 
    #     "func_type": "series",
    #     "column": "data",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # # NLP tests using db_agent_single (3)
    # {
    #     "name": "imdb_db_agent",
    #     "table": "imdb_vector_test",
    #     "func_type": "nlp",
    #     "column": "comment_vec",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # {
    #     "name": "financial_phrasebank_db_agent",
    #     "table": "financial_phrasebank_vector_test",
    #     "func_type": "nlp",
    #     "column": "comment_vec", 
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # {
    #     "name": "nlp_db_agent",
    #     "table": "nlp_vector_test",
    #     "func_type": "nlp",
    #     "column": "comment_vec",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # Image tests using db_agent_single (3)
    # {
    #     "name": "cifar_db_agent",
    #     "table": "cifar_image_vector_table",
    #     "func_type": "image_classification",
    #     "column": "image_vector",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # {
    #     "name": "stanford_dogs_db_agent",
    #     "table": "stanford_dogs_image_vector_table", 
    #     "func_type": "image_classification",
    #     "column": "image_vector",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # },
    # {
    #     "name": "imagenet_db_agent",
    #     "table": "imagenet_image_vector_table",
    #     "func_type": "image_classification", 
    #     "column": "image_vector",
    #     "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    # }
]

import json
from datetime import datetime

def run_single_slice_test(use_new_queries=False):
    # Row counts to iterate through
    ROW_COUNTS = [1000, 2000, 5000, 10000]
    
    # Select which queries to use
    if use_new_queries:
        QUERIES_TO_USE = NEW_SQL_QUERIES
        queries_name = "NEW_SQL_QUERIES (db_agent_single)"
    else:
        QUERIES_TO_USE = SQL_QUERIES
        queries_name = "SQL_QUERIES (predict_batch_float8)"
    
    print(f"Using {queries_name} for testing")
    
    # Store all results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "queries_used": queries_name,
        "tests": []
    }
    
    for sql_info in QUERIES_TO_USE:
        print(f"\n{'='*80}")
        print(f"Testing SQL: {sql_info['name']}")
        print(f"Table: {sql_info['table']}")
        # Show either model or func_type depending on the query type
        if 'model' in sql_info:
            print(f"Model: {sql_info['model']}")
        if 'func_type' in sql_info:
            print(f"Function Type: {sql_info['func_type']}")
        print(f"Column: {sql_info['column']}")
        print(f"{'='*80}")
        
        test_results = {
            "name": sql_info["name"],
            "table": sql_info["table"],
            "results": []
        }
        
        # Add model or func_type to test_results for clarity
        if 'model' in sql_info:
            test_results["model"] = sql_info["model"]
        if 'func_type' in sql_info:
            test_results["func_type"] = sql_info["func_type"]
        test_results["column"] = sql_info["column"]
        
        for row_count in ROW_COUNTS:
            print(f"\n  Testing with {row_count} rows...")
            
            # Format the query with actual values
            if use_new_queries:
                # For NEW_SQL_QUERIES, first format the outer placeholders, then the row_count
                formatted_query_template = sql_info['query'].format(
                    func_type=sql_info['func_type'],
                    column=sql_info['column'],
                    table=sql_info['table']
                )
                # Replace the row_count placeholder - note: the template uses {{row_count}} which becomes {row_count} after first format
                formatted_query = formatted_query_template.format(row_count=row_count)
            else:
                # For original SQL_QUERIES, format with model, symbol, column, table, row_count
                formatted_query = sql_info['query'].format(
                    model=sql_info['model'],
                    symbol='cpu',
                    column=sql_info['column'],
                    table=sql_info['table'],
                    row_count=row_count
                )
            
            print(f"  Query: {formatted_query}")
            
            result = run_single_task_worker(0, row_count, 1, 'cpu', formatted_query)
            
            # Add metadata to result
            result['row_count'] = row_count
            result['formatted_query'] = formatted_query
            
            print(f"  Result: {result}")
            
            # Add to test results
            test_results["results"].append(result)
        
        # Add test results to all results
        all_results["tests"].append(test_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY OF ALL TESTS - {queries_name}")
    print(f"{'='*80}")
    
    for test in all_results["tests"]:
        print(f"\nTest: {test['name']} (Table: {test['table']})")
        for result in test["results"]:
            if result["status"] == "success":
                print(f"  Rows: {result['row_count']}, Latency: {result['latency']:.4f}s")
            else:
                print(f"  Rows: {result['row_count']}, Status: {result['status']}, Error: {result.get('error', 'Unknown')}")
    
    # Write results to file with different names based on query type
    filename = f"single_slice_test_results_{'new' if use_new_queries else 'original'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filename}")

def run_both_tests():
    """Run both sets of queries"""
    print("Running original SQL_QUERIES...")
    run_single_slice_test(use_new_queries=False)
    
    print("\n" + "="*80)
    print("Running NEW_SQL_QUERIES...")
    run_single_slice_test(use_new_queries=True)

if __name__ == "__main__":
    run_single_slice_test(True)
