#!/usr/bin/env python3
"""
Skew Dataset Performance Test for PGDL/MorphingDB.

Creates synthetic datasets with different class skew ratios (A vs B),
then measures latency of predict_batch_float8 and db_agent_single
on each skewed dataset. Tests 3 series tables across 6 skew ratios.

Usage:
    python skew_dataset_test.py
"""

import psycopg2
import time
from morphingdb_test.config import db_config
import json
import csv
from datetime import datetime


def create_skew_dataset(table_name, source_table, skew_column, skew_ratios, total_rows=None):
    """
    Create a skew dataset based on different ratios.
    
    Args:
        table_name: Name of the new skew table to create
        source_table: Source table to copy data from (e.g., 'slice_test')
        skew_column: Column to use for creating skew (e.g., 'id' or a computed value)
        skew_ratios: List of tuples like [(50, 'A'), (50, 'B')] meaning 50% A, 50% B
        total_rows: Total number of rows to create, if None use all from source
    
    Returns:
        Number of rows created
    """
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    try:
        cur.execute(f"drop table if exists {table_name};")
        
        cur.execute(f"""
            create table {table_name} as 
            select *, 0 as skew_class 
            from {source_table} 
            {'limit ' + str(total_rows) if total_rows else ''};
        """)
        
        cur.execute(f"select count(*) from {table_name};")
        row_count = cur.fetchone()[0]
        
        if skew_ratios and len(skew_ratios) >= 2:
            cur.execute(f"alter table {table_name} add column skew_label text;")
            
            b_percentage = skew_ratios[1][0]
            cur.execute(f"update {table_name} set skew_label = 'B' where ctid in (select ctid from {table_name} limit {int(row_count * b_percentage / 100)});")
            cur.execute(f"update {table_name} set skew_label = 'A' where skew_label is null;")
        
        conn.commit()
        print(f"Created skew dataset '{table_name}' with {row_count} rows, ratios: {skew_ratios}")
        return row_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error creating skew dataset: {e}")
        raise
    finally:
        conn.close()


def get_table_row_count(table_name):
    """Get the number of rows in a table."""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute(f"select count(*) from {table_name};")
    count = cur.fetchone()[0]
    conn.close()
    return count


def run_prediction_test(table_name, model, column, symbol='cpu', query_times=1):
    """
    Test predict_batch_float8 on entire table (no LIMIT).
    """
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        sql = f"select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table_name};"
        
        start_time = time.time()
        cur.execute("select register_process();")
        
        return_rows = 0
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall()
            return_rows += len(result)
        
        end_time = time.time()
        conn.close()
        
        return {
            "status": "success",
            "latency": end_time - start_time,
            "return_rows": return_rows,
            "query_times": query_times
        }
    except Exception as e:
        print(f"Error in prediction test: {e}")
        return {"status": "failed", "error": str(e)}


def run_db_agent_single_test(table_name, func_type, column, query_times=1):
    """
    Test db_agent_single on entire table (no LIMIT).
    """
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        sql = f"select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table_name}) AS sub_table;"
        
        start_time = time.time()
        cur.execute("select register_process();")
        
        return_rows = 0
        for i in range(query_times):
            cur.execute(sql)
            result = cur.fetchall()
            return_rows += len(result)
        
        end_time = time.time()
        conn.close()
        
        return {
            "status": "success",
            "latency": end_time - start_time,
            "return_rows": return_rows,
            "query_times": query_times
        }
    except Exception as e:
        print(f"Error in db_agent_single test: {e}")
        return {"status": "failed", "error": str(e)}


SKEW_TEST_CONFIG = {
    "slice_test": {
        "source_table": "slice_test",
        "model": "slice",
        "func_type": "series",
        "column": "data",
        "default_skew_ratios": [(95, 'A'), (5, 'B')],
        "base_table_size": None
    },
    "swarm_test": {
        "source_table": "swarm_test", 
        "model": "swarm",
        "func_type": "series",
        "column": "data",
        "default_skew_ratios": [(95, 'A'), (5, 'B')],
        "base_table_size": None
    },
    "year_predict_test": {
        "source_table": "year_predict_test",
        "model": "year_predict", 
        "func_type": "series",
        "column": "data",
        "default_skew_ratios": [(95, 'A'), (5, 'B')],
        "base_table_size": None
    }
}


def run_skew_dataset_test():
    """
    Main function to run skew dataset performance tests.
    Tests predict_batch_float8 and db_agent_single on datasets with different skew ratios.
    """
    print("Starting Skew Dataset Performance Test")
    print("=" * 80)
    
    skew_ratios_list = [
        [(50, 'A'), (50, 'B')],
        [(70, 'A'), (30, 'B')],
        [(80, 'A'), (20, 'B')],
        [(90, 'A'), (10, 'B')],
        [(95, 'A'), (5, 'B')],
        [(99, 'A'), (1, 'B')],
    ]
    
    query_times_list = [1, 5, 10]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "skew_dataset_performance",
        "description": "Performance test on datasets with different skew ratios (no LIMIT, full table scan)",
        "skew_ratios_tested": skew_ratios_list,
        "query_times_tested": query_times_list,
        "results": []
    }
    
    for table_name, config in SKEW_TEST_CONFIG.items():
        print(f"\n{'='*80}")
        print(f"Testing table: {table_name}")
        print(f"Model: {config['model']}, FuncType: {config['func_type']}")
        print(f"Column: {config['column']}")
        print(f"{'='*80}")
        
        base_row_count = get_table_row_count(config["source_table"])
        print(f"Source table row count: {base_row_count}")
        
        table_results = {
            "table_name": table_name,
            "model": config["model"],
            "func_type": config["func_type"],
            "column": config["column"],
            "source_row_count": base_row_count,
            "skew_tests": []
        }
        
        for skew_ratios in skew_ratios_list:
            skew_table_name = f"{table_name}_skew_{skew_ratios[0][0]}_{skew_ratios[1][0]}"
            print(f"\n  Creating skew table: {skew_table_name}")
            print(f"  Skew ratios: {skew_ratios}")
            
            row_count = create_skew_dataset(
                table_name=skew_table_name,
                source_table=config["source_table"],
                skew_column="skew_label",
                skew_ratios=skew_ratios,
                total_rows=None
            )
            
            skew_test_result = {
                "skew_ratio": f"{skew_ratios[0][0]}-{skew_ratios[1][0]}",
                "actual_row_count": row_count,
                "predict_batch_results": [],
                "db_agent_single_results": []
            }
            
            for query_times in query_times_list:
                print(f"\n    Testing with query_times={query_times}")
                
                print(f"    Running predict_batch_float8...")
                predict_result = run_prediction_test(
                    table_name=skew_table_name,
                    model=config["model"],
                    column=config["column"],
                    symbol='cpu',
                    query_times=query_times
                )
                predict_result["query_times"] = query_times
                skew_test_result["predict_batch_results"].append(predict_result)
                print(f"    predict_batch_float8 result: {predict_result}")
                
                print(f"    Running db_agent_single...")
                db_agent_result = run_db_agent_single_test(
                    table_name=skew_table_name,
                    func_type=config["func_type"],
                    column=config["column"],
                    query_times=query_times
                )
                db_agent_result["query_times"] = query_times
                skew_test_result["db_agent_single_results"].append(db_agent_result)
                print(f"    db_agent_single result: {db_agent_result}")
            
            table_results["skew_tests"].append(skew_test_result)
        
        all_results["results"].append(table_results)
    
    print_summary(all_results)
    save_results(all_results)
    return all_results


def print_summary(all_results):
    """Print a summary table of the test results."""
    print(f"\n{'='*80}")
    print("SUMMARY OF SKEW DATASET PERFORMANCE TESTS")
    print(f"{'='*80}")
    
    print(f"\n{'Table':<25} {'Skew Ratio':<12} {'Query Times':<12} {'Predict (s)':<15} {'DBAgent (s)':<15}")
    print("-" * 80)
    
    for table_result in all_results["results"]:
        table_name = table_result["table_name"]
        for skew_test in table_result["skew_tests"]:
            ratio = skew_test["skew_ratio"]
            for i, (predict_res, db_agent_res) in enumerate(zip(
                    skew_test["predict_batch_results"], 
                    skew_test["db_agent_single_results"])):
                qt = predict_res["query_times"]
                predict_latency = f"{predict_res['latency']:.4f}" if predict_res["status"] == "success" else "FAILED"
                db_agent_latency = f"{db_agent_res['latency']:.4f}" if db_agent_res["status"] == "success" else "FAILED"
                print(f"{table_name:<25} {ratio:<12} {qt:<12} {predict_latency:<15} {db_agent_latency:<15}")


def save_results(all_results):
    """Save results to JSON and CSV files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_filename = f"skew_test_results_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved to: {json_filename}")
    
    csv_filename = f"skew_test_results_{timestamp}.csv"
    save_results_to_csv(all_results, csv_filename)
    print(f"CSV results saved to: {csv_filename}")


def save_results_to_csv(all_results, csv_filename):
    """Convert JSON results to CSV format."""
    header = ['Table', 'Model', 'FuncType', 'SourceRows', 'SkewRatio', 'QueryTimes', 
              'PredictBatch_Latency', 'PredictBatch_Status', 
              'DBAgentSingle_Latency', 'DBAgentSingle_Status']
    
    rows = []
    for table_result in all_results["results"]:
        for skew_test in table_result["skew_tests"]:
            for i in range(len(skew_test["predict_batch_results"])):
                predict_res = skew_test["predict_batch_results"][i]
                db_agent_res = skew_test["db_agent_single_results"][i]
                
                row = [
                    table_result["table_name"],
                    table_result["model"],
                    table_result["func_type"],
                    table_result["source_row_count"],
                    skew_test["skew_ratio"],
                    predict_res["query_times"],
                    predict_res.get("latency", "N/A") if predict_res["status"] == "success" else "FAILED",
                    predict_res["status"],
                    db_agent_res.get("latency", "N/A") if db_agent_res["status"] == "success" else "FAILED",
                    db_agent_res["status"]
                ]
                rows.append(row)
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    run_skew_dataset_test()