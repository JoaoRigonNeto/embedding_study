from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

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

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

print(f"Cosine simliarity from ({sentences[0]}): ")
print(cosine_similarity([embeddings[0]], embeddings[1:]))
print(f"Cosine distances from ({sentences[0]}): ")
print(cosine_distances([embeddings[0]], embeddings[1:]))