import psycopg2
import time
from concurrent.futures import ProcessPoolExecutor
from morphingdb_test.config import db_config
import json
import csv
import sys
from collections import defaultdict


def json_to_csv(json_file_path, csv_file_path):
    """
    Convert JSON test results to CSV format.
    
    Args:
        json_file_path (str): Path to the input JSON file
        csv_file_path (str): Path to the output CSV file
    """
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all unique row counts and sort them
    row_counts = set()
    for test in data['tests']:
        for result in test['results']:
            row_counts.add(result['row_count'])
    
    row_counts = sorted(list(row_counts))
    
    # Prepare header
    header = ['Test Name', 'Table', 'Model/FuncType', 'Column'] + [f'Latency_{rc}' for rc in row_counts]
    
    # Prepare rows
    rows = []
    for test in data['tests']:
        # Determine the model or function type
        model_or_func = test.get('model', test.get('func_type', 'N/A'))
        
        # Initialize row with test metadata
        row = [test['name'], test['table'], model_or_func, test.get('column', 'N/A')]
        
        # Create a mapping of row_count to latency for this test
        latency_map = {}
        for result in test['results']:
            if result['status'] == 'success':
                latency_map[result['row_count']] = result['latency']
            else:
                latency_map[result['row_count']] = 'ERROR'
        
        # Add latency values for each row count (or 'N/A' if not available)
        for rc in row_counts:
            latency_value = latency_map.get(rc, 'N/A')
            row.append(latency_value)
        
        rows.append(row)
    
    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Successfully converted {json_file_path} to {csv_file_path}")
    print(f"Found {len(data['tests'])} tests and {len(row_counts)} different row counts: {row_counts}")

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
        start_time = time.time()
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
    {
        "name": "swarm_db_agent", 
        "table": "swarm_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "year_predict_db_agent",
        "table": "year_predict_test", 
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    # # NLP tests using db_agent_single (3)
    {
        "name": "imdb_db_agent",
        "table": "imdb_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "financial_phrasebank_db_agent",
        "table": "financial_phrasebank_vector_test",
        "func_type": "nlp",
        "column": "comment_vec", 
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "nlp_db_agent",
        "table": "nlp_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    # Image tests using db_agent_single (3)
    {
        "name": "cifar_db_agent",
        "table": "cifar_image_vector_table",
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "stanford_dogs_db_agent",
        "table": "stanford_dogs_image_vector_table", 
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "imagenet_db_agent",
        "table": "imagenet_image_vector_table",
        "func_type": "image_classification", 
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    }
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
            
            # Determine the appropriate symbol based on test type
            if use_new_queries:
                # For NEW_SQL_QUERIES, use func_type to determine symbol
                if sql_info['func_type'] == 'series':
                    symbol = 'cpu'
                else:  # nlp or image_classification
                    symbol = 'gpu'
            else:
                # For original SQL_QUERIES, use test name to determine symbol
                if sql_info['name'] in ['slice_predict', 'swarm_predict', 'year_predict_test']:
                    symbol = 'cpu'
                else:
                    symbol = 'gpu'
                
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
                    symbol=symbol,
                    column=sql_info['column'],
                    table=sql_info['table'],
                    row_count=row_count
                )
            
            print(f"  Query: {formatted_query}")
            
            result = run_single_task_worker(0, row_count, 1, symbol, formatted_query)
            
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
    filename = f"single_test_results_{'new' if use_new_queries else 'original'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    json_to_csv(filename, filename.replace('.json', '.csv'))
    print(f"\nResults saved to: {filename}")

def run_both_tests():
    """Run both sets of queries"""
    print("Running original SQL_QUERIES...")
    run_single_slice_test(use_new_queries=False)
    
    print("\n" + "="*80)
    print("Running NEW_SQL_QUERIES...")
    run_single_slice_test(use_new_queries=True)

if __name__ == "__main__":
    run_both_tests()
    # run_single_slice_test(True)
