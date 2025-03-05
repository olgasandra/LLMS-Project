# %%
# Importing the necessary libraries
import openai
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
from embedding_functions import get_embedding

# %%
# Set up directories
EMBEDDINGS_DIR = Path.cwd().parent.parent.joinpath("data/embeddings")
OUTPUTS_DIR = Path.cwd().parent.parent.joinpath("data/outputs")
PROCESSED_DIR = Path.cwd().parent.parent.joinpath("data/processed")
TEST_DIR = Path.cwd().parent.parent.joinpath("data/test_data")

# %%
client = openai

# %%
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
client.api_key = os.getenv("API_KEY")

# %%
# Accessing the openai api_key
# REMOVE?
#client.api_key = "API_Key"

# %%
# Importing the preprocessed data
preprocessed_isco_08_data = pd.read_csv(PROCESSED_DIR / "preprocessed_isco_08_data.csv")

# %%
# Creating the Embeddings in the dataset
preprocessed_isco_08_data['embeddings'] = preprocessed_isco_08_data["detailed_explanation"].apply(get_embedding)

# saving the embedded data
preprocessed_isco_08_data.to_csv(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.csv", index = False)
preprocessed_isco_08_data.to_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.pkl")


# %%
# Importing the embedded data
isco_08_df_with_embeddings = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.pkl")

# retrieving the first few rows of embeddings column
isco_08_df_with_embeddings['embeddings'].head()


