from opensearchpy import OpenSearch, helpers
import os
import numpy as np
from dotenv import load_dotenv
from time import sleep
from sentence_transformers import SentenceTransformer

load_dotenv()
host = 'localhost'
port = 9200
auth = ('admin', os.environ.get("OPENSEARCH_PASSWORD")) 


client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_auth=auth,
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)
index_name = 'my-embeddings-index'

if client.ping():
    print("Connection ok")
else:
    print("Connection not established succesfully")
    exit(0)



#--------------------------------------------------Generate Embeddings--------------------------------------------------
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

texts = [
    "bright sun shines",
    "gentle wind blows",
    "birds sing sweetly",
    "trees sway softly",
    "waves crash gently",
    "stars twinkle brightly",
    "flowers bloom vividly",
    "rivers flow calmly",
    "thoughts drift freely",
    "dreams unfold slowly",
    "time slips away",
    "words paint pictures",
    "hearts beat strongly",
    "days pass quickly",
    "nights fall silently",
    "rain falls softly",
    "colors blend beautifully",
    "mountains stand tall",
    "oceans stretch wide",
    "clouds float lazily",
    "leaves rustle gently",
    "paths wind onward",
    "memories linger sweetly",
    "smiles light faces",
    "laughter fills rooms",
    "stories unfold richly",
    "moments pass fleetingly",
    "shadows lengthen slowly",
    "fires burn warmly",
    "candles glow softly",
    "music plays sweetly",
    "feelings run deep",
    "hopes rise high",
    "journeys start small",
    "secrets stay hidden",
    "answers come slowly",
    "questions hang heavy",
    "choices shape futures",
    "changes come suddenly",
    "seasons turn gracefully",
    "words speak volumes",
    "silence holds meaning",
    "breezes cool gently",
    "sunsets fade beautifully",
    "moonlight shines softly",
    "stars gleam distantly",
    "footsteps echo softly",
    "dreams seem real",
    "wishes fly high",
    "moments feel timeless"
]


def gen_embds(texts, model) -> list:
    embds = model.encode(texts)
    return [item.tolist() for item in embds]
embeddings = gen_embds(texts, model)
zipped_embeddings = dict(zip(texts, embeddings))
#--------------------------------------------------Single insert--------------------------------------------------
# document = {
#     "text_content": texts[0],
#     "embeddings": embeddings[0]
# }

# response = client.index(index_name, document)
# print(response)
#--------------------------------------------------Bulk insert--------------------------------------------------
# documents = [
#     {"_index": index_name, "_source":{"text_content":key , "embeddings": value}}  
#     for key, value in zipped_embeddings.items()
# ]

# response = helpers.bulk(client, documents)
# print(response)

#--------------------------------------------------Query k nearest neighbors--------------------------------------------------
query_text = "moments feel timeless"
query_vector = gen_embds([query_text], model)[0]
k = 3
print(query_vector)

query = {
    "size": k,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": k
            }
        },
    }
}

response = client.search(index=index_name, body=query)
print(query_text)
print(response)

for hit in response["hits"]["hits"]:
    print(f"ID: {hit['_id']}, Score: {hit['_score']}, Source: {hit['_source']['text_content']}")

#--------------------------------------------------Create index--------------------------------------------------
# index_body = {
# "settings": {
#     "index.knn": True
#   },
#   "mappings": {
#     "properties": {
#       "embedding": {
#         "type": "knn_vector",
#         "dimension": 384,
#         "space_type": "cosinesimil",
#         "method": {
#           "name": "hnsw",
#           "engine": "faiss",
#           "parameters": {
#             "ef_construction": 128,
#             "m": 24
#           }
#         }
#       },
#       "text_content": {
#         "type": "text"
#       }
#     }
#   }
# }
# try:
#     if not client.indices.exists(index=index_name):
#         response = client.indices.create(index_name, body=index_body)
#         print("Index created:")
#         print(response)
#     else:
#         print(f"Index '{index_name}' already exists.")
# except Exception as e:
#     print(f"An error occurred: {e}")

#--------------------------------------------------Delete index--------------------------------------------------
# try:
#     if client.indices.exists(index=index_name):
#         response = client.indices.delete(index_name)
#         print(response)
#     else:
#         print(f"Index '{index_name}' already exists.")
# except Exception as e:
#     print(f"An error occurred: {e}")
