# %% [markdown]
# **EVALUATION FOR RAG SYSTEM**

# %%
# Importing the necessary libraries
import ast
import pandas as pd
import os
import openai
from pathlib import Path
from dotenv import load_dotenv

from ragas import evaluate
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import context_recall, faithfulness, answer_correctness

# %%
# Set directories
TEST_DIR = Path.cwd().joinpath("data/test_data")
OUTPUTS_DIR = Path.cwd().joinpath("data/outputs")

# %%
client = openai

# %%
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
client.api_key = os.getenv("OPENAI_API_KEY")

# %%
# Load the test_dataset
test_data = pd.read_csv(TEST_DIR / "RAG_test_dataset_1_5_rows.csv")

# Convert retrieved_docs from string to list (if needed)
test_data["retrieved_docs"] = test_data["retrieved_docs"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


# %%
# Check if 'generated_answer', 'retrieved_docs' and 'ground_truth' have NaNs and replace
test_data['generated_answer'] = test_data['generated_answer'].fillna("No response").astype(str)
test_data['retrieved_docs'] = test_data['retrieved_docs'].apply(lambda x: x if isinstance(x, list) else ["No retrieved documents"])
test_data['ground_truth'] = test_data['ground_truth'].fillna("No reference available").astype(str)

test_data.head()

# %% [markdown]
# ### Evaluation
# Metrics: (context_recall, faithfulness, answer_correctness)

# %%
# time
# Create SingleTurnSample instances for each row in the dataset
samples = []

for i, row in test_data.iterrows():
    sample = SingleTurnSample(
        user_input=row['query'],
        retrieved_contexts=row['retrieved_docs'],
        response=row['generated_answer'],
        reference=row['ground_truth'],
    )
    samples.append(sample)

# Create an EvaluationDataset
evaluation_dataset = EvaluationDataset(samples=samples)

# Evaluate using Ragas
results = evaluate(
    evaluation_dataset,
    metrics=[context_recall, faithfulness, answer_correctness]
)

# Saving the evaluation results to a csv
evaluation_result_df = results.to_pandas()
evaluation_result_df.to_csv(OUTPUTS_DIR / 'evaluation_result_1_5_rows_df.csv')

# Display the results
print(results)


