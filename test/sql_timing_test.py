#!/usr/bin/env python3
"""
Simple script to test execution time of SQL statements.
"""

import psycopg2
from morphingdb_test.config import db_config
import time


def test_sql_execution_time(sql_query, iterations=1):
    """
    Test the execution time of a given SQL statement.
    
    Args:
        sql_query (str): The SQL query to test
        iterations (int): Number of times to execute the query (for average timing)
    
    Returns:
        dict: Execution statistics including average time, min time, max time, etc.
    """
    print(f"Testing SQL Query: {sql_query}")
    print(f"Iterations: {iterations}")
    print("-" * 80)
    
    # Connect to the database
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None
    
    execution_times = []
    total_rows = 0
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}:")
        
        # Execute the query
        start_time = time.time()
        try:
            # Optional: register process if needed (as in original)
            cur.execute("select register_process();")
            cur.execute(sql_query)
            result = cur.fetchall()
            conn.commit()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            row_count = len(result)
            total_rows += row_count
            
            print(f"  ✓ SUCCESS - Returned {row_count} rows in {execution_time:.4f}s")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"  ✗ FAILED - Execution time: {execution_time:.4f}s, Error: {str(e)}")
    
    # Calculate statistics
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        stats = {
            'query': sql_query,
            'iterations': iterations,
            'execution_times': execution_times,
            'average_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_rows': total_rows // iterations if iterations > 0 else 0  # Average rows
        }
        
        print("\n" + "="*80)
        print("EXECUTION STATISTICS:")
        print(f"Avg Time: {avg_time:.4f}s")
        print(f"Min Time: {min_time:.4f}s")
        print(f"Max Time: {max_time:.4f}s")
        print(f"Total Rows (avg): {total_rows // iterations if iterations > 0 else 0}")
        print("="*80)
        
        # Close the connection
        conn.close()
        
        return stats
    else:
        conn.close()
        return None


def main():
    """Main function to demonstrate SQL timing test."""
    print("SQL Execution Time Testing Tool")
    print("="*80)
    
    # Example SQL query - user can modify this
    sample_query = "SELECT * FROM slice_test LIMIT 10;"
    
    # Get SQL query from user input or use default
    user_query = input(f"Enter SQL query to test (or press Enter for default '{sample_query}'): ").strip()
    if not user_query:
        user_query = sample_query
    
    # Get number of iterations
    try:
        iterations_input = input("Enter number of iterations (or press Enter for 1): ").strip()
        iterations = int(iterations_input) if iterations_input else 1
    except ValueError:
        iterations = 1
        print("Invalid input, using 1 iteration")
    
    # Test the SQL query
    stats = test_sql_execution_time(user_query, iterations)
    
    if stats:
        print(f"\nTest completed successfully!")
    else:
        print(f"\nTest failed!")


if __name__ == "__main__":
    main()