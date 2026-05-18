#!/usr/bin/env python3
"""Standalone year_predict dataset importer - no morphingdb_test package dependency."""
import os
import time
import pandas as pd
import psycopg2

DB_CONFIG = {
    "dbname": "postgres", "host": "localhost", "port": "5432",
    "user": "why", "password": "123456"
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "morphingdb_test", "data", "series", "yead_predict", "YearPredictionMSD.csv")

print("Starting year_predict import...")
start = time.time()
dataframe = pd.read_csv(CSV_PATH)
value_columns = dataframe.columns[1:91]

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# mvec table
cur.execute("DROP TABLE IF EXISTS year_predict_test;")
cur.execute("CREATE TABLE year_predict_test (data mvec, res float4);")
for index, row in dataframe.iterrows():
    values = row[value_columns].tolist()
    values_str = '[' + ', '.join(str(v) for v in values) + ']'
    cur.execute("INSERT INTO year_predict_test VALUES('{}', {});".format(values_str+'{1,90}', row['value0']))
    if index % 5000 == 0:
        conn.commit()
conn.commit()
print("  year_predict mvec done: {:.1f}s ({}) rows".format(time.time()-start, len(dataframe)))

# origin table
value_columns_all = dataframe.columns.tolist()
cur.execute("DROP TABLE IF EXISTS year_predict_origin_test;")
create_sql = 'CREATE TABLE year_predict_origin_test (\n  '
for i, col in enumerate(value_columns_all):
    create_sql += '"{}" float4,\n'.format(col)
create_sql = create_sql.rstrip(',\n') + '\n);'
cur.execute(create_sql)
cur.execute("COPY year_predict_origin_test FROM '{}' WITH (FORMAT csv, HEADER true, DELIMITER ',');".format(CSV_PATH))
conn.commit()
print("  year_predict origin done: {:.1f}s".format(time.time()-start))
conn.close()
print("Year_predict import complete.")
