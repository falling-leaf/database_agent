#!/usr/bin/env python3
"""
Script to convert JSON test results to CSV format.
Each row represents a different test type.
Each column represents different row counts with latency values.
"""

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


def main():
    if len(sys.argv) != 3:
        print("Usage: python json_to_csv_converter.py <input_json_file> <output_csv_file>")
        print("Example: python json_to_csv_converter.py single_slice_test_results.json results.csv")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_csv = sys.argv[2]
    
    json_to_csv(input_json, output_csv)


if __name__ == "__main__":
    # If run as a standalone script, use command line args
    if len(sys.argv) > 1:
        main()
    else:
        # Otherwise, use the specific file mentioned in the request
        input_file = "/home/why/pgdl/test/single_slice_test_results_new_20260201_125425.json"
        output_file = "/home/why/pgdl/test/single_slice_test_results_new_20260201_125425.csv"
        json_to_csv(input_file, output_file)