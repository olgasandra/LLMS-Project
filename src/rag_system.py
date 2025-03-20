# Importing the necessary libraries
import openai
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os
from generation_functions import query_system, generate_response

# %%
# Set up directories
EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")

# %%
client = openai

# %%
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
api_key = os.getenv("OPENAI_API_KEY")

# %% [markdown]
# ### Testing the RAG System

# %%
# Importing embeddings data
embeddings_data = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings_for_translation.pkl")


# write a query
description = "logging climber"

# Retrieve the most relevant row
response = query_system(description, isco_08_df_pkl)

print(response)



# # %% [markdown]
# # **RAG SYSTEM**

# # %%
# # Importing the necessary libraries
# import openai
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from dotenv import load_dotenv
# import os
# from generation_functions import query_system, generate_response

# # %%
# # Set up directories
# EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")

# # %%
# client = openai

# # %%
# # Load environment variables from the .env file
# load_dotenv()

# # Access environment variables as if they came from the actual environment
# api_key = os.getenv("OPENAI_API_KEY")

# # %% [markdown]
# # ### Testing the RAG System

# # %%
# # Importing embeddings data
# embeddings_data = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.pkl")

# # %%
# query = "This person works as a waiter at restaurant?"

# # Retrieve the most relevant row
# retrieved_row = query_system(query, embeddings_data)

# # Generate the response
# response = generate_response(query, retrieved_row)
# print(response)


