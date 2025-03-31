# script with functions for embedding

# import packages
import openai
import numpy as np

client = openai

# Function to generate embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    """_summary_

    Args:
        text (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Replace newlines with spaces to avoid errors
    text = text.replace("\n", " ")
    # Calling the OpenAI API to create embeddings
    response = client.Embedding.create(input=text, model=model)
    # Return the embedding vector
    return response['data'][0]['embedding']

# Function to calculate similarity (dot product)
def calculate_similarity(query_embedding, db_embeddings):
    """_summary_

    Args:
        query_embedding (_type_): _description_
        db_embeddings (_type_): _description_

    Returns:
        _type_: _description_
    """    
    similarities = np.dot(db_embeddings, query_embedding)
    return similarities
