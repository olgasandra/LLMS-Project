# %% [markdown]
# **DASH APP**

# %%
# Importing the necessary libraries
import re
import numpy as np
import pandas as pd
import openai
from pathlib import Path
from dotenv import load_dotenv
import os
from dash import Dash, dcc, html, Input, Output, State
from IPython.display import Image, display
from generation_functions import query_system, generate_response

# %%
# Set up directories
EMBEDDINGS_DIR = Path.cwd().parent.parent.joinpath("data/embeddings")
CONTENT_DIR = Path.cwd().parent.parent.joinpath("content")

# %%
client = openai

# %%
# Accessing the openai api_key
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
api_key = os.getenv("API_KEY")

# %%
# --------------------- #
#   DISPLAY IMAGE IN NOTEBOOK
# --------------------- #

image_path = CONTENT_DIR / "PNG_IN_NOTEBOOK_NISR_02_1.png"
display(Image(filename=image_path))

# %%
# --------------------- #
#   LOAD DATASET
# --------------------- #

df = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.pkl")

# %%
# --------------------- #
#   DASH APP SETUP
# --------------------- #

app = Dash(__name__)

app.layout = html.Div(
    style={
        "backgroundColor": "#343541",
        "minHeight": "100vh",
        "padding": "20px",
        "boxSizing": "border-box"
    },
    children=[
        # App Title
        html.Div(
            "Get the code for your occupation",
            style={
                "textAlign": "center",
                "fontSize": "2em",
                "color": "#ffffff",
                "marginBottom": "20px"
            }
        ),

        # Input Box and Submit Button
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "40px"
            },
            children=[
                dcc.Input(
                    id="query-input",
                    type="text",
                    placeholder="Type your occupation query...",
                    style={
                        "width": "60%",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "border": "1px solid #ccc",
                        "marginRight": "10px"
                    }
                ),
                html.Button(
                    "Get Code",
                    id="get-code-btn",
                    n_clicks=0,
                    style={
                        "padding": "10px 20px",
                        "backgroundColor": "#083e7f",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "fontWeight": "bold"
                    }
                )
            ]
        ),

        # Response Output Section
        html.Div(
            id="response-output",
            style={
                "width": "80%",
                "margin": "0 auto",
                "textAlign": "left"
            }
        )
    ]
)

# --------------------- #
#   DASH CALLBACK: Handle User Input
# --------------------- #

@app.callback(
    Output("response-output", "children"),
    Input("get-code-btn", "n_clicks"),
    State("query-input", "value"),
)
def update_output(n_clicks, query):
    """
    When user clicks 'Get Code', retrieve similar job descriptions,
    pass them to GPT, extract the occupation code, and display it.
    """
    if n_clicks > 0 and query:
        retrieved_rows = query_system(query, df)
        response_text = generate_response(query, retrieved_rows)

        # Extract ISCO Code and Occupation from GPT Response
        match = re.search(r"The occupational code is:\s*([\w\d]+)\s+for\s+(.*)\.", response_text)

        if match:
            isco_code = match.group(1)
            occupation = match.group(2)

            # Display Result in a Styled Card
            return html.Div(
                style={
                    "backgroundColor": "#444654",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "10px"
                },
                children=[
                    html.Div(
                        isco_code,
                        style={
                            "fontWeight": "bold",
                            "fontSize": "1.2em",
                            "color": "#fff",
                            "marginRight": "20px"
                        }
                    ),
                    html.Div(
                        occupation,
                        style={
                            "color": "#fff",
                            "fontSize": "1em"
                        }
                    )
                ]
            )
        else:
            # Display GPT's raw response if format is unexpected
            return html.Div(
                response_text,
                style={
                    "backgroundColor": "#444654",
                    "color": "#fff",
                    "borderRadius": "12px",
                    "padding": "15px",
                    "marginBottom": "10px"
                }
            )

    # Default empty response before any query
    return html.Div(" ", style={"color": "#fff"})

# --------------------- #
#   RUN DASH SERVER
# --------------------- #

if __name__ == "__main__":
    app.run_server(debug=True)


# %%



