# %% [markdown]
# **RAG SYSTEM**

# %%
# Importing the necessary libraries
import openai
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

# %%
from occ_coder.generation.generation_functions import RAG_query_system

# %%
# Set up directories
DATA_DIR = Path.cwd().joinpath("data")
EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")
OUTPUTS_DIR = Path.cwd().joinpath("data/outputs")
PROCESSED_DIR = Path.cwd().joinpath("data/processed")
TEST_DIR = Path.cwd().joinpath("data/test_data")

# %%
client = openai

# %%
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
api_key = os.getenv("API_KEY")

# %% [markdown]
# ### Testing the RAG System

# %%
# Importing embeddings data
embeddings_data = pd.read_pickle(EMBEDDINGS_DIR.joinpath("isco_08_df_with_embeddings.pkl"))

# %%
query = "This person works as a waiter at restaurant?"

# Importing embeddings data
embeddings_data = pd.read_pickle(EMBEDDINGS_DIR.joinpath("isco_08_df_with_embeddings.csv"))

# Generate the response
description = "This person works as a waiter at restaurant?"

RAG_query_system(description, embeddings_data, model="text-embedding-ada-002")


