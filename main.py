from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import mteb


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

model_names = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/all-distilroberta-v1',
    'sentence-transformers/multi-qa-mpnet-base-dot-v1',
]

for model_name in model_names:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)

    print(f"\nModel: {model_name}")
    print(f"Cosine similarity from ({sentences[0]}): ")
    print(cosine_similarity([embeddings[0]], embeddings[1:]))
    print(f"Cosine distances from ({sentences[0]}): ")
    print(cosine_distances([embeddings[0]], embeddings[1:]))

    tasks = mteb.get_tasks(tasks=[
        "Banking77Classification", #classification banking dataset test:
        "STSBenchmark", #Semantic similarity test
        ])
    evaluation = mteb.MTEB(tasks=tasks)

    results = evaluation.run(model, output_folder=f"results/{model_name.split(sep='/')[-1]}")
