import openai
import numpy as np
import pandas as pd
from langdetect import detect
import os
from dotenv import load_dotenv
import openai
from occ_coder.embedding.embedding_functions import get_embedding, calculate_similarity

# Load environment variables from the .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for OpenAI
openai.api_key = api_key

# Function to translate text to English (detects language automatically)
def gpt_translate_to_english(text, source_lang="auto"):
    """Translate text to English using GPT."""
    if source_lang == "auto":
        try:
            source_lang = detect(text)
        except:
            source_lang = "Unknown"
    
    system_prompt = f"You are a professional translator. Translate the following {source_lang} text to English accurately."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Translation failed: {e}"

# Function to summarize the provided text
def gpt_summarize_text(text, language="English"):
    """Summarize the provided text using GPT."""
    try:
        system_prompt = f"You are an expert summarizer. Summarize the following text in clear and concise {language}."
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=30
        )
        
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    
    except Exception as e:
        return f"Summarization failed: {e}"

# Function to generate response for a specific question based on retrieved rows (with only occupation code output)
def RAG_generate_response(question, retrieved_rows):
    """Generate only code response based on the retrieved rows."""
    combined_context = "\n\n".join(retrieved_rows['ISCO_n_TITLE'].dropna())

    if not combined_context.strip():
        return "No such code for the occupation."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant trained to classify job descriptions and provide occupation codes based on context. "
                    "Always respond with code only. "
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

# Function to generate response for a specific question based on retrieved rows (with occupation code and title output)
def app_generate_response(question, retrieved_rows):
    """Generate a response that includes occupation code and title based on retrieved rows."""
    combined_context = "\n\n".join(retrieved_rows['ISCO_n_TITLE'].dropna())

    if not combined_context.strip():
        return "No such code for the occupation."

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant trained to classify job descriptions and provide occupation codes based on context. "
                    "Always respond with: 'The occupational code is: [code] for [ISCO_n_TITLE]. "
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

# Function to query the system, retrieve top 5 similar results, and generate only a code response based on similarity
def RAG_query_system(description, df, model="text-embedding-ada-002"):
    """Query the system, retrieve the top 5 most similar results, and generate a response."""
    try:
        lang = detect(description)
        if lang != 'en':
            description = gpt_translate_to_english(description, lang)
    except:
        pass  # If language detection fails, proceed with original text

    description = gpt_summarize_text(description)

    query_embedding = get_embedding(description, model=model)
    db_embeddings = np.vstack(df['embeddings'])
    similarities = calculate_similarity(query_embedding, db_embeddings)

    top_5_indices = np.argsort(similarities)[-5:][::-1]
    retrieved_rows = df.iloc[top_5_indices]

    response = RAG_generate_response(description, retrieved_rows)

    return response

# Function to query the system, retrieve top 5 similar results, and generate response with occupation code, title and confidence level
def app_query_system(description, df, model="text-embedding-ada-002"):
    """Query the system, retrieve top 5 similar results, and generate response with occupation code, title and confidence level."""
    try:
        lang = detect(description)
        if lang != 'en':
            description = gpt_translate_to_english(description, lang)
    except:
        pass  # If language detection fails, proceed with original text

    description = gpt_summarize_text(description)

    # Generate the embedding for the user query
    query_embedding = get_embedding(description, model=model)
    
    # Get the embeddings for the dataset
    db_embeddings = np.vstack(df['embeddings'])
    
    # Calculate similarity between the query and the dataset
    similarities = calculate_similarity(query_embedding, db_embeddings)

    # Get the indices of the top 5 most similar rows
    top_5_indices = np.argsort(similarities)[-5:][::-1]
    retrieved_rows = df.iloc[top_5_indices]

    # Use the generate_response function to get the top occupation based on the retrieved rows
    top_occupation_response = app_generate_response(description, retrieved_rows)

    # Calculate confidence levels based on similarity scores
    results = []
    for i, idx in enumerate(top_5_indices):  # Iterate over top_5_indices, not retrieved_rows.iterrows()
        similarity_score = similarities[idx]
        confidence_level = round(similarity_score * 100, 2)  # Confidence as a percentage
        row = df.iloc[idx]  # Access the row by the correct index
        results.append({
            "ISCO_n_TITLE": row['ISCO_Key_words'],
            "ISCO_08": row['ISCO_08'],  # Use ISCO_08 instead of ISCO_code
            "confidence": f"{confidence_level}%",
            "similarity": similarity_score
        })

    # The top occupation should be the one generated by GPT using the top 5 rows
    top_occupation = top_occupation_response
                    
    # Remove the top occupation from the results to display other occupations
    other_occupations = results

    return top_occupation, other_occupations