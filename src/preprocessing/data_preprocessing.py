# %% [markdown]
# **DATA PREPROCESSING**

# %%
# Importing the necessary libraries
import pandas as pd
from pathlib import Path

# %%
# Set directories
DATA_DIR = Path.cwd().joinpath("data")
PROCESSED_DIR = Path.cwd().joinpath("data/processed")

# %%
# Importing the dataset
isco_08_df = pd.read_csv(DATA_DIR / "ISCO-08-Data.csv")

# Number of rows and columns
isco_08_df.shape

# %%
# Replacing spaces with underscores in column names
isco_08_df.columns = isco_08_df.columns.str.replace(' ', '_')
isco_08_df.columns

# %%
# Filter rows where the column "ISCO_08_Code" contains exactly 4 digits
isco_08_df = isco_08_df[isco_08_df['ISCO_08_Code'].astype(str).str.match(r'^\d{4}$')]
isco_08_df.head()

# %%
# Combining all columns into "detailed_explanation"
isco_08_df["detailed_explanation"] = isco_08_df.apply(
    lambda row: "\n".join(
        [
            f"{col}:\n{row[col]}"
            for col in isco_08_df.columns if col != "detailed_explanation"
        ]
    ),
    axis=1
)

isco_08_df.head()

# %%
# Saving the preprocessed data
isco_08_df.to_csv(PROCESSED_DIR / "preprocessed_isco_08_data.csv", index=False)


