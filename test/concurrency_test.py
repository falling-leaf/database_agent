import psycopg2
import time
from concurrent.futures import ProcessPoolExecutor
from morphingdb_test.config import db_config
import json
import csv
from collections import defaultdict
from datetime import datetime

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
        # The sql_query should already be formatted with all necessary parameters
        sql = sql_query
        return_rows = 0
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall() # 确保数据读取完毕
            return_rows += len(result)
        expected_rows = row_count * query_times
        print(f"Task {task_id}: return_rows: {return_rows}, expected_rows: {expected_rows}")
        
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

def test_concurrency(concurrency_level, row_count, query_times, symbol, sql_query):
    """
    Test a specific concurrency level
    """
    print(f"Testing concurrency level: {concurrency_level}")
    
    # Use ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=concurrency_level) as executor:
        # Submit tasks
        futures = []
        for i in range(concurrency_level):
            future = executor.submit(run_single_task_worker, i, row_count, query_times, symbol, sql_query)
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=60)  # 60 second timeout
                results.append(result)
            except Exception as e:
                print(f"Timeout or error in task {i}: {str(e)}")
                results.append({"task_id": i, "status": "failed", "error": f"Timeout or error: {str(e)}"})
    
    # Count success and failure rates
    successful_tasks = [r for r in results if r["status"] == "success"]
    failed_tasks = [r for r in results if r["status"] == "failed"]
    
    success_rate = len(successful_tasks) / len(results) if len(results) > 0 else 0
    failure_rate = len(failed_tasks) / len(results) if len(results) > 0 else 0
    
    print(f"Concurrency level {concurrency_level}: {len(successful_tasks)}/{len(results)} succeeded, Success rate: {success_rate:.2%}")
    
    test_result = {
        "concurrency_level": concurrency_level,
        "total_tasks": len(results),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "results": results,
        "row_count_per_task": row_count,
        "query_times_per_task": query_times,
        "sql_query": sql_query
    }
    
    return test_result

def find_max_concurrency(start_level=1, max_level=50, increment=5, row_count=1000, query_times=10, symbol='cpu', sql_query=None):
    """
    Find the maximum concurrency level that can run without failures
    """
    print(f"Starting concurrency stress test from {start_level} to {max_level} with increment {increment}")
    
    all_results = []
    max_safe_concurrency = start_level - 1  # Start with the level before the test begins
    
    for concurrency_level in range(start_level, max_level + 1, increment):
        test_result = test_concurrency(concurrency_level, row_count, query_times, symbol, sql_query)
        all_results.append(test_result)
        
        # If there are failures, we've exceeded the safe limit
        if test_result["failed_tasks"] > 0:
            print(f"Failures detected at concurrency level {concurrency_level}. Maximum safe level appears to be {max_safe_concurrency}.")
            break
        else:
            # Update max safe concurrency if this level succeeded completely
            max_safe_concurrency = concurrency_level
        
        # Small delay between tests to allow system to recover
        time.sleep(2)
    
    # If we reached max_level without failures, do binary search to find exact limit
    if all(result["failed_tasks"] == 0 for result in all_results[-1:]):  # Last test passed
        print(f"All tests up to {max_level} passed. Performing binary search to find exact limit.")
        low = max_safe_concurrency
        high = max_level
        
        while low < high:
            mid = (low + high + 1) // 2
            test_result = test_concurrency(mid, row_count, query_times, symbol, sql_query)
            all_results.append(test_result)
            
            if test_result["failed_tasks"] == 0:
                low = mid  # This level works
                max_safe_concurrency = mid
            else:
                high = mid - 1  # This level fails, try lower
            
            time.sleep(1)  # Brief pause between binary search steps
    
    return all_results, max_safe_concurrency

# Define SQL queries to test - 9 test cases across 3 types
ORIGINAL_SQL_QUERIES = [
    # Series tests (3) - using CPU
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
    # NLP tests (3) - using GPU
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
    # Image tests (3) - using GPU
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

# New SQL queries based on the commented examples using db_agent_single
NEW_SQL_QUERIES = [
    # Series tests using db_agent_single (3) - using CPU
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
    # NLP tests using db_agent_single (3) - using GPU
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
    # Image tests using db_agent_single (3) - using GPU
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

def run_concurrency_test():
    """
    Main function to run the concurrency test with both original and new queries
    """
    # Test both original and new queries
    all_test_results = {}
    
    print("Starting Concurrency Stress Tests for Both Original and New Queries")
    print("="*80)
    
    # Test original queries
    print("\nTesting ORIGINAL SQL QUERIES (predict_batch_float8)...")
    original_results = run_concurrency_tests_for_queries(ORIGINAL_SQL_QUERIES, "ORIGINAL")
    all_test_results["original"] = original_results
    
    # Test new queries
    print("\nTesting NEW SQL QUERIES (db_agent_single)...")
    new_results = run_concurrency_tests_for_queries(NEW_SQL_QUERIES, "NEW")
    all_test_results["new"] = new_results
    
    # Store all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results_dict = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "concurrency_stress_test_both_query_types",
        "original_queries_results": original_results,
        "new_queries_results": new_results
    }
    
    # Print summary
    print("="*80)
    print("CONCURRENCY STRESS TEST SUMMARY")
    print("="*80)
    print(f"Timestamp: {all_results_dict['timestamp']}")
    
    # Write results to file
    filename = f"concurrency_test_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filename}")
    
    # Also create CSV version
    json_to_csv(filename, filename.replace('.json', '.csv'))
    
    return all_test_results

def run_concurrency_tests_for_queries(queries, query_type):
    """
    Run concurrency tests for a specific set of queries across different row counts
    """
    results_by_query = {}
    
    # Row counts to iterate through
    ROW_COUNTS = [1000, 2000, 5000, 10000]
    
    for sql_info in queries:
        print(f"\n{'='*60}")
        print(f"Testing {query_type} SQL: {sql_info['name']}")
        print(f"Table: {sql_info['table']}")
        # Show either model or func_type depending on the query type
        if 'model' in sql_info:
            print(f"Model: {sql_info['model']}")
        if 'func_type' in sql_info:
            print(f"Function Type: {sql_info['func_type']}")
        print(f"Column: {sql_info['column']}")
        print(f"{'='*60}")
        
        # Determine the appropriate symbol based on test type
        if query_type == "NEW":
            # For NEW_SQL_QUERIES, use func_type to determine symbol
            if sql_info['func_type'] == 'series':
                symbol = 'cpu'
            else:  # nlp or image_classification
                symbol = 'gpu'
        else:
            # For ORIGINAL_SQL_QUERIES, use test name to determine symbol
            if sql_info['name'] in ['slice_predict', 'swarm_predict', 'year_predict_test']:
                symbol = 'cpu'
            else:
                symbol = 'gpu'
        
        # Store results for all row counts
        query_results_by_row_count = {}
        
        for row_count in ROW_COUNTS:
            print(f"\n  Testing with {row_count} rows...")
            
            # Format the query with actual values for this row count
            if query_type == "NEW":
                # For NEW_SQL_QUERIES, first format the outer placeholders, then the row_count
                formatted_query_template = sql_info['query'].format(
                    func_type=sql_info['func_type'],
                    column=sql_info['column'],
                    table=sql_info['table']
                )
                # Replace the row_count placeholder - note: the template uses {{row_count}} which becomes {row_count} after first format
                formatted_query = formatted_query_template.format(row_count=row_count)
            else:
                # For ORIGINAL_SQL_QUERIES, format with model, symbol, column, table, row_count
                formatted_query = sql_info['query'].format(
                    model=sql_info['model'],
                    symbol=symbol,
                    column=sql_info['column'],
                    table=sql_info['table'],
                    row_count=row_count
                )
            
            print(f"  Query: {formatted_query}")
            print(f"  Device: {symbol}")
            
            # Run the concurrency test for this specific query and row count
            all_results, max_safe_concurrency = find_max_concurrency(
                start_level=1,
                max_level=16,  # Start with reasonable upper bound for individual query test
                increment=2,
                row_count=row_count,
                query_times=5,
                symbol=symbol,
                sql_query=formatted_query
            )
            
            # Store results for this row count
            query_results_by_row_count[row_count] = {
                "max_safe_concurrency": max_safe_concurrency,
                "test_parameters": {
                    "row_count_per_task": row_count,
                    "query_times_per_task": 5,
                    "sql_query": formatted_query
                },
                "results": all_results
            }
        
        # Store all results for this query across all row counts
        results_by_query[sql_info['name']] = {
            "name": sql_info["name"],
            "table": sql_info["table"],
            "symbol": symbol,
            "results_by_row_count": query_results_by_row_count
        }
    
    return results_by_query

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
    
    # Prepare header for concurrency test results
    header = ['Test Type', 'Query Name', 'Row Count', 'Concurrency Level', 'Total Tasks', 'Successful Tasks', 'Failed Tasks', 
              'Success Rate', 'Failure Rate', 'Symbol (Device)']
    
    # Prepare rows
    rows = []
    
    # Check if this is the new format with original and new queries and results by row count
    if 'original_queries_results' in data and 'new_queries_results' in data:
        # Process original queries results
        for query_name, query_data in data['original_queries_results'].items():
            if 'results_by_row_count' in query_data:
                for row_count, row_count_data in query_data['results_by_row_count'].items():
                    if 'results' in row_count_data:
                        for result in row_count_data['results']:
                            row = [
                                'ORIGINAL',  # Test Type
                                query_name,  # Query Name
                                row_count,  # Row Count
                                result['concurrency_level'],
                                result['total_tasks'],
                                result['successful_tasks'],
                                result['failed_tasks'],
                                f"{result['success_rate']:.4f}",
                                f"{result['failure_rate']:.4f}",
                                query_data.get('symbol', 'N/A')  # Device symbol
                            ]
                            rows.append(row)
        
        # Process new queries results
        for query_name, query_data in data['new_queries_results'].items():
            if 'results_by_row_count' in query_data:
                for row_count, row_count_data in query_data['results_by_row_count'].items():
                    if 'results' in row_count_data:
                        for result in row_count_data['results']:
                            row = [
                                'NEW',  # Test Type
                                query_name,  # Query Name
                                row_count,  # Row Count
                                result['concurrency_level'],
                                result['total_tasks'],
                                result['successful_tasks'],
                                result['failed_tasks'],
                                f"{result['success_rate']:.4f}",
                                f"{result['failure_rate']:.4f}",
                                query_data.get('symbol', 'N/A')  # Device symbol
                            ]
                            rows.append(row)
    else:
        # Handle legacy format or other format that has direct 'results'
        if 'results' in data:
            for result in data['results']:
                row = [
                    'LEGACY',  # Test Type
                    'N/A',  # Query Name
                    'N/A',  # Row Count
                    result['concurrency_level'],
                    result['total_tasks'],
                    result['successful_tasks'],
                    result['failed_tasks'],
                    f"{result['success_rate']:.4f}",
                    f"{result['failure_rate']:.4f}",
                    'N/A'  # Device symbol
                ]
                rows.append(row)
    
    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Successfully converted {json_file_path} to {csv_file_path}")
    print(f"Processed {len(rows)} concurrency test result entries")


if __name__ == "__main__":
    run_concurrency_test()