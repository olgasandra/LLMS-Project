# Importing the necessary libraries
import openai
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
from generation_functions import RAG_query_system

EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")


# Importing embeddings data
embeddings_data = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings_for_translation.pkl")


# write a query
description = "akora akazi ko kubaka amazu"

# generating a response of the query
response = RAG_query_system(description, embeddings_data)

print(response)
