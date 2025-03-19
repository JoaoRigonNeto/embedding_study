import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(
    dbname="mydatabase",
    user="joao",
    password="123",
    host="localhost"
)
table_name = "test_vector"

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
sentences = [
    "The cat sat on the mat.",
    "A feline rested upon the rug.",
    "Dogs bark loudly at night.",
    "Canines vocalize during the dark.",
    "I enjoy eating apples.",
    "Consuming the fruit is pleasing.",
    "The sky is blue today.",
    "Today, the heavens are azure.",
    "She went to the store.",
    "She visited the market."
]

def create_vector_table(conn, table_name, vector_size) -> None:
    with conn.cursor() as cur:
        cur.execute(f" CREATE TABLE {table_name} ( id serial PRIMARY KEY, embeddings vector({vector_size})); ")
        conn.commit()

def insert_values_vector_table(conn, table_name, embeddings):
    with conn.cursor() as cur:
        execute_values(cur, f"INSERT INTO {table_name} (embeddings) VALUES %s", [(v.tolist(),) for v in embeddings])
        conn.commit()


create_vector_table(conn=conn, table_name=table_name, vector_size=384)

model = SentenceTransformer(model_name)
embeddings = model.encode(sentences)

insert_values_vector_table(conn=conn, table_name=table_name, embeddings=embeddings)

# QUERY VALUES export PGPASSWORD=123 ;psql -h localhost -U joao -d mydatabase -p 5432 -c "SELECT * FROM test_vector;"

conn.close()


