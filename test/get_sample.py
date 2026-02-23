#!/usr/bin/env python3
"""
Script to execute all NEW_SQL_QUERIES commands in order, removing the limit row_count part.
"""

import psycopg2
from morphingdb_test.config import db_config
import time


# New SQL queries based on the commented examples
NEW_SQL_QUERIES = [
    # Series tests using db_agent_single (3)
    {
        "name": "slice_db_agent",
        "table": "slice_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "swarm_db_agent", 
        "table": "swarm_test",
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "year_predict_db_agent",
        "table": "year_predict_test", 
        "func_type": "series",
        "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    # # NLP tests using db_agent_single (3)
    {
        "name": "imdb_db_agent",
        "table": "imdb_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "financial_phrasebank_db_agent",
        "table": "financial_phrasebank_vector_test",
        "func_type": "nlp",
        "column": "comment_vec", 
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "nlp_db_agent",
        "table": "nlp_vector_test",
        "func_type": "nlp",
        "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    # Image tests using db_agent_single (3)
    {
        "name": "cifar_db_agent",
        "table": "cifar_image_vector_table",
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "stanford_dogs_db_agent",
        "table": "stanford_dogs_image_vector_table", 
        "func_type": "image_classification",
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    },
    {
        "name": "imagenet_db_agent",
        "table": "imagenet_image_vector_table",
        "func_type": "image_classification", 
        "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit 5000) AS sub_table;"
    }
]


def execute_new_sql_queries():
    """Execute all NEW_SQL_QUERIES commands in order, removing the limit row_count part."""
    
    print("Starting execution of NEW_SQL_QUERIES...")
    print("="*80)
    
    # Connect to the database
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return
    
    # Execute each query in order
    for i, sql_info in enumerate(NEW_SQL_QUERIES):
        print(f"\n[{i+1}/{len(NEW_SQL_QUERIES)}] Executing: {sql_info['name']}")
        print(f"Table: {sql_info['table']}")
        print(f"Function Type: {sql_info['func_type']}")
        print(f"Column: {sql_info['column']}")
        
        # Format the query with actual values (without row_count since it's removed)
        formatted_query = sql_info['query'].format(
            func_type=sql_info['func_type'],
            column=sql_info['column'],
            table=sql_info['table']
        )
        
        print(f"Query: {formatted_query}")
        
        # Execute the query
        start_time = time.time()
        try:
            cur.execute("select register_process();")  # Register the process as in the original
            cur.execute(formatted_query)
            result = cur.fetchall()
            conn.commit()
            end_time = time.time()
            
            execution_time = end_time - start_time
            row_count = len(result)
            
            print(f"✓ SUCCESS - Returned {row_count} rows in {execution_time:.4f}s")
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"✗ FAILED - Execution time: {execution_time:.4f}s, Error: {str(e)}")
    
    # Close the connection
    conn.close()
    print("\n" + "="*80)
    print("Finished executing all NEW_SQL_QUERIES")


if __name__ == "__main__":
    execute_new_sql_queries()