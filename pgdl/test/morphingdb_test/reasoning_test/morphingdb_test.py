import psycopg2
import time
import json
import re
import os
from morphingdb_test.config import (
    db_config,
    reasoning_cross_encoder_output_path,
    reasoning_deberta_output_path,
    reasoning_flant5_output_path
)
from .import_dataset import (
    import_reasoning_dataset,
    import_reasoning_mvec_dataset,
    REASONING_TABLE,
    REASONING_VECTOR_TABLE,
    SAMPLE_DATA
)

ROW_COUNT_LIST = [1, 10, 50, 100]

REASONING_TEST_FILE = 'result/reasoning_test.json'
REASONING_VECTOR_TEST_FILE = 'result/reasoning_vector_test.json'
timing_data = []
timing_data_vector = []


def parse_timing_info(timing_str):
    pattern = r'total: (\d+) ms\n load model: (\d+) ms\((\d+\.\d+)%\)\n pre process: (\d+) ms\((\d+\.\d+)%\)\n infer: (\d+) ms\((\d+\.\d+)%\)\n post process: (\d+) ms\((\d+\.\d+)%\)'
    match = re.search(pattern, timing_str)
    if match:
        return {
            'total_time': int(match.group(1)),
            'load_model_time': int(match.group(2)),
            'load_model_percent': float(match.group(3)),
            'pre_time': int(match.group(4)),
            'pre_percent': float(match.group(5)),
            'infer_time': int(match.group(6)),
            'infer_percent': float(match.group(7)),
            'post_time': int(match.group(8)),
            'post_percent': float(match.group(9)),
        }
    return None


def create_models():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    models = [
        ('cross_encoder', reasoning_cross_encoder_output_path),
        ('deberta_reader', reasoning_deberta_output_path),
        ('flan_t5_reader', reasoning_flant5_output_path),
    ]

    for model_name, model_path in models:
        cur.execute("select * from model_info where model_name = %s;", (model_name,))
        res = cur.fetchall()
        if len(res) == 0:
            cur.execute("select create_model(%s, %s, '', '');", (model_name, model_path))

    conn.commit()
    conn.close()
    print("Reasoning models registered.")


def reasoning_init_data():
    import_reasoning_dataset()
    print("Import reasoning dataset done")
    import_reasoning_mvec_dataset()
    print("Import reasoning mvec dataset done")


def reasoning_vector_test(limit_flag: str, symbol: str = 'gpu'):
    for count in ROW_COUNT_LIST:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("select register_process();")
        start = time.time()

        sql = "select predict_batch_float8('flan_t5_reader', '{}', reasoning_vec) over (rows between current row and 31 following) from {} limit {};".format(
            symbol, REASONING_VECTOR_TABLE, count)
        print(f"Executing: {sql}")
        cur.execute(sql)
        end = time.time()
        cur.execute("select print_cost();")
        res = parse_timing_info(cur.fetchall()[0][0])
        res['scan_time'] = (end - start) * 1000000 - res['total_time']
        res['total_time'] = (end - start) * 1000000
        conn.close()

        try:
            with open(REASONING_VECTOR_TEST_FILE.format(limit_flag), 'r') as f_vector:
                timing_data_vector = json.load(f_vector)
        except (FileNotFoundError, json.JSONDecodeError):
            timing_data_vector = []

        timing_data_vector.append({
            "sql": sql,
            "count": count,
            "total_time": res["total_time"] / 1000000,
            "scan_time": res["scan_time"] / 1000000,
            "load_model_time": res["load_model_time"] / 1000000,
            "pre_time": res["pre_time"] / 1000000,
            "infer_time": res["infer_time"] / 1000000,
            "post_time": res["post_time"] / 1000000
        })

        with open(REASONING_VECTOR_TEST_FILE.format(limit_flag), 'w') as f_vector:
            json.dump(timing_data_vector, f_vector, indent=4)


def reasoning_single_test():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    cur.execute("select register_process();")

    sample = SAMPLE_DATA[0]
    sql = "select predict_batch_float8('flan_t5_reader', 'gpu', reasoning_vec) over (rows between current row and 31 following) from reasoning_vector_test where id = 1;"
    print(f"Single sample test: {sql}")

    start = time.time()
    cur.execute(sql)
    results = cur.fetchall()
    end = time.time()

    cur.execute("select print_cost();")
    res = parse_timing_info(cur.fetchall()[0][0])
    conn.close()

    print(f"Result: {results}")
    print(f"Total time: {(end - start) * 1000:.2f} ms")
    if res:
        print(f"  Load model: {res['load_model_time']} ms ({res['load_model_percent']}%)")
        print(f"  Pre process: {res['pre_time']} ms ({res['pre_percent']}%)")
        print(f"  Infer: {res['infer_time']} ms ({res['infer_percent']}%)")
        print(f"  Post process: {res['post_time']} ms ({res['post_percent']}%)")


def reasoning_all_test():
    create_models()
    print('Create models done')
    reasoning_vector_test('', 'gpu')
    print('Reasoning vector test done')
    reasoning_single_test()
    print('Reasoning single test done')


if __name__ == "__main__":
    reasoning_all_test()
