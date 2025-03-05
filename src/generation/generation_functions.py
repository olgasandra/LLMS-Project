import numpy as np
import openai

# script with functions for generative

# Retrieving system function to find and retrieve the top 4 results
def query_system(question, df, model="text-embedding-ada-002"):
    """_summary_

    Args:
        question (_type_): _description_
        df (_type_): _description_
        model (str, optional): _description_. Defaults to "text-embedding-ada-002".

    Returns:
        _type_: _description_
    """    
    # Generate embedding for the query
    query_embedding = get_embedding(question, model=model)

    # Calculate similarities
    db_embeddings = np.vstack(df['embeddings'])  # Convert list of embeddings to matrix
    similarities = calculate_similarity(query_embedding, db_embeddings)

    # Retrieve indices of the top 4 most similar rows
    top_4_indices = np.argsort(similarities)[-4:][::-1]  # Sort, get last 4, and reverse to descending order
    retrieved_rows = df.iloc[top_4_indices]

    return retrieved_rows


# Response Generation system function using GPT-3.5 Turbo
def generate_response(question, retrieved_rows):
    """_summary_

    Args:
        question (_type_): _description_
        retrieved_rows (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Combine the top 4 contexts into a single context string
    combined_context = "\n\n".join(retrieved_rows['detailed_explanation'].dropna())

    # Check if the combined context is valid (not empty)
    if not combined_context.strip():
        # If no valid context is found, return a direct response without querying GPT
        return "No such code for the occupation."

    # Use GPT-3.5 Turbo to generate a response based on the combined context
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant trained to classify job descriptions and provide occupation codes based on context. "
                    "Always respond with the occupational code only, without any additional text or explanation. "
                    "If the context does not match the query, respond with 'NULL'."
                ),
            },
            {
                "role": "user",
                "content": f"Based on these contexts:\n\n{combined_context}\n\nAnswer the query: {question}"
            }
        ]
    )

    return response['choices'][0]['message']['content']
