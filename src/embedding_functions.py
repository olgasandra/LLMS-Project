import openai
import numpy as np

# Function to generate embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

# Function to calculate similarity (dot product)
def calculate_similarity(query_embedding, db_embeddings):
    similarities = np.dot(db_embeddings, query_embedding)
    return similarities
