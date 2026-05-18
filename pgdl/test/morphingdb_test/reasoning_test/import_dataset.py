import psycopg2
from morphingdb_test.config import db_config, spiece_model_path

REASONING_TABLE = 'reasoning_test'
REASONING_VECTOR_TABLE = 'reasoning_vector_test'

SAMPLE_DATA = [
    {
        "context": "The quick brown fox jumps over the lazy dog.",
        "question": "What does the fox jump over?",
        "answer": "the lazy dog"
    },
    {
        "context": "Python is a high-level programming language known for its readability.",
        "question": "What is Python known for?",
        "answer": "readability"
    },
    {
        "context": "The Earth revolves around the Sun in approximately 365 days.",
        "question": "How many days does it take Earth to revolve around the Sun?",
        "answer": "365"
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil?",
        "answer": "100 degrees Celsius"
    },
    {
        "context": "The Great Wall of China is over 13,000 miles long.",
        "question": "How long is the Great Wall of China?",
        "answer": "13,000 miles"
    },
    {
        "context": "Photosynthesis converts sunlight into chemical energy in plants.",
        "question": "What does photosynthesis convert sunlight into?",
        "answer": "chemical energy"
    },
    {
        "context": "The human body has 206 bones in adulthood.",
        "question": "How many bones does an adult human have?",
        "answer": "206"
    },
    {
        "context": "DNA stands for deoxyribonucleic acid.",
        "question": "What does DNA stand for?",
        "answer": "deoxyribonucleic acid"
    },
    {
        "context": "The speed of light is approximately 299,792,458 meters per second.",
        "question": "What is the speed of light?",
        "answer": "299,792,458 meters per second"
    },
    {
        "context": "Mount Everest is the highest mountain above sea level at 8,849 meters.",
        "question": "How high is Mount Everest?",
        "answer": "8,849 meters"
    },
]


def import_reasoning_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("create table if not exists " + REASONING_TABLE + " (id serial primary key, context text, question text, answer text);")
    conn.commit()

    cur.execute("delete from " + REASONING_TABLE + ";")
    conn.commit()

    for item in SAMPLE_DATA:
        sql = f"INSERT INTO {REASONING_TABLE} (context, question, answer) VALUES ('{item['context'].replace(chr(39), chr(39)+chr(39))}', '{item['question'].replace(chr(39), chr(39)+chr(39))}', '{item['answer'].replace(chr(39), chr(39)+chr(39))}')"
        cur.execute(sql)
        conn.commit()

    conn.close()
    print(f"Imported {len(SAMPLE_DATA)} rows into {REASONING_TABLE}")


def import_reasoning_mvec_dataset():
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute("create table if not exists " + REASONING_VECTOR_TABLE + " (id serial primary key, reasoning_vec mvec);")
    conn.commit()

    cur.execute("delete from " + REASONING_VECTOR_TABLE + ";")
    conn.commit()

    for item in SAMPLE_DATA:
        context_escaped = item['context'].replace(chr(39), chr(39)+chr(39))
        question_escaped = item['question'].replace(chr(39), chr(39)+chr(39))
        sql = f"INSERT INTO {REASONING_VECTOR_TABLE} (reasoning_vec) VALUES (text_to_vector('{spiece_model_path}', '{context_escaped} || {question_escaped}'))"
        cur.execute(sql)
        conn.commit()

    conn.close()
    print(f"Imported {len(SAMPLE_DATA)} rows into {REASONING_VECTOR_TABLE}")


if __name__ == "__main__":
    import_reasoning_dataset()
    import_reasoning_mvec_dataset()
