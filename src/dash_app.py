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
from generation_functions import query_system, generate_response, gpt_translate_to_english, gpt_summarize_text

# %%
# Set up directories
EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")
CONTENT_DIR = Path.cwd().joinpath("content")

# %%
client = openai

# %%
# Accessing the openai api_key
# Load environment variables from the .env file
load_dotenv()

# Access environment variables as if they came from the actual environment
api_key = os.getenv("OPENAI_API_KEY")


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App Layout with inline CSS styles
app.layout = html.Div([
    html.Div([
        html.Img(
            src=CONTENT_DIR/'PNG_IN_NOTEBOOK_NISR_02_1.png', 
            style={'width': '100%', 'height': 'auto', 'display': 'block'}
        ),
        html.Hr(),
    ], style={'padding': '0', 'margin': '0'}),
    
    # Tabs
    dcc.Tabs(id="tabs", value='home', children=[
        dcc.Tab(label='Home', value='home'),
        dcc.Tab(label='Single Query', value='query-classification'),
        dcc.Tab(label='Batch Processing', value='batch-processing'),
        dcc.Tab(label='Quality Checking', value='quality-checking'),
    ], style={'fontFamily': 'Poppins', 'fontWeight': 'bold', 'color': '#34495e'}),

    # Content for the selected tab
    html.Div(id='tab-content', style={'marginTop': '20px'}),
], style={
    'fontFamily': 'Poppins, sans-serif', 
    'backgroundColor': '#ffffff', 
    'padding': '0', 
    'margin': '0',
    'fontSize': '25px'
})

# Tab content callback
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tab_content(tab_name):
    if tab_name == 'home':
        return html.H3("Welcome to the Home Tab", style={'textAlign': 'center', 'color': '#34495e'})
    
    elif tab_name == 'query-classification':
        return html.Div([
            html.H3("Enter your query to classify an occupation:", style={'textAlign': 'center', 'color': '#34495e', 'fontFamily': 'Poppins', 'fontWeight': 'normal'}),
            
            # Input Text Area for Query (Centered)
            html.Div([
                dcc.Textarea(
                    id='query-textarea',
                    placeholder='Type your occupation query here...',
                    style={
                        'width': '100%', 'height': 100, 'resize': 'none', 'padding': '10px', 
                        'borderRadius': '5px', 'border': '1px solid #ccc', 'margin': '0 auto'
                    }
                ),
                # Submit icon button
                html.Button(
                    html.I(className='fa fa-envelope', style={'fontSize': '24px', 'color': 'white'}),  # Font Awesome send icon
                    id='submit-button', 
                    n_clicks=0, 
                    style={
                        'position': 'absolute', 'top': '50%', 'right': '10px', 'transform': 'translateY(-50%)', 
                        'backgroundColor': '#2980b9', 'border': 'none', 'borderRadius': '50%', 'width': '40px', 
                        'height': '40px', 'cursor': 'pointer'
                    }
                )
            ], style={'position': 'relative', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'width': '80%', 'margin': '0 auto'}),
            
            html.Br(),
            html.Div(id='classification-result', style={'textAlign': 'center', 'color': '#34495e', 'marginTop': '20px'}),
            html.Div(id='other-occupations', style={'textAlign': 'center', 'color': '#34495e'})
        ])
    
    elif tab_name == 'batch-processing':
        return html.H3("Batch Processing Tab", style={'textAlign': 'center', 'color': '#34495e'})
    
    elif tab_name == 'quality-checking':
        return html.H3("Quality Checking Tab", style={'textAlign': 'center', 'color': '#34495e'})

# Callback to handle the query submission
@app.callback(
    [Output('classification-result', 'children'),
     Output('other-occupations', 'children')],
    [Input('submit-button', 'n_clicks')],
    [Input('query-textarea', 'value')]
)
def classify_occupation(n_clicks, query):
    if n_clicks > 0:
        if query:
            # Load your dataframe with embeddings
            df = pd.read_pickle(EMBEDDINGS_DIR/'isco_08_df_with_embeddings_for_translation.pkl')  # Load your preprocessed occupation data
            top_occupation, other_occupations = query_system(query, df)  # Call the function from rag_functions.py
            
            # Display the top occupation
            top_occupation_str = html.H5(top_occupation, style= {'textAlign': 'left', 'color': '#34495e', 'fontFamily': 'Poppins', 'fontWeight': 'normal'})  # You can modify this if needed
            
            # Create the table for other occupations
            other_occupations_table = html.Table(
                [html.Thead(html.Tr([html.Th('Occupation Title'), html.Th('ISCO_08'), html.Th('Confidence')]))]
                + [html.Tbody([html.Tr([html.Td(row['ISCO_n_TITLE']), html.Td(row['ISCO_08']), html.Td(row['confidence'])]) for row in other_occupations])]
            )
            
            # Title for the other occupations section
            other_occupations_title = html.H4("Other Potential Occupations", style={'textAlign': 'left', 'color': '#34495e', 'fontFamily': 'Poppins', 'fontWeight': 'bold'})
            
            return top_occupation_str, html.Div([other_occupations_title, other_occupations_table])
        else:
            return html.H5("Please enter a query to classify.", style={'color': '#e74c3c', 'fontFamily': 'Poppins', 'fontWeight': 'normal'}), ""
    return "", ""

# Run the app
if __name__ == '__main__':
    app.run(debug=True)







# # %% [markdown]
# # **DASH APP**

# # %%
# # Importing the necessary libraries
# import re
# import numpy as np
# import pandas as pd
# import openai
# from pathlib import Path
# from dotenv import load_dotenv
# import os
# from dash import Dash, dcc, html, Input, Output, State
# from IPython.display import Image, display
# from generation_functions import query_system, generate_response

# # %%
# # Set up directories
# EMBEDDINGS_DIR = Path.cwd().joinpath("data/embeddings")
# CONTENT_DIR = Path.cwd().joinpath("content")

# # %%
# client = openai

# # %%
# # Accessing the openai api_key
# # Load environment variables from the .env file
# load_dotenv()

# # Access environment variables as if they came from the actual environment
# api_key = os.getenv("API_KEY")

# # %%
# # --------------------- #
# #   DISPLAY IMAGE IN NOTEBOOK
# # --------------------- #

# image_path = CONTENT_DIR / "PNG_IN_NOTEBOOK_NISR_02_1.png"
# display(Image(filename=image_path))

# # %%
# # --------------------- #
# #   LOAD DATASET
# # --------------------- #

# df = pd.read_pickle(EMBEDDINGS_DIR / "isco_08_df_with_embeddings.pkl")

# # %%
# # --------------------- #
# #   DASH APP SETUP
# # --------------------- #

# app = Dash(__name__)

# app.layout = html.Div(
#     style={
#         "backgroundColor": "#343541",
#         "minHeight": "100vh",
#         "padding": "20px",
#         "boxSizing": "border-box"
#     },
#     children=[
#         # App Title
#         html.Div(
#             "Get the code for your occupation",
#             style={
#                 "textAlign": "center",
#                 "fontSize": "2em",
#                 "color": "#ffffff",
#                 "marginBottom": "20px"
#             }
#         ),

#         # Input Box and Submit Button
#         html.Div(
#             style={
#                 "display": "flex",
#                 "justifyContent": "center",
#                 "marginBottom": "40px"
#             },
#             children=[
#                 dcc.Input(
#                     id="query-input",
#                     type="text",
#                     placeholder="Type your occupation query...",
#                     style={
#                         "width": "60%",
#                         "padding": "10px",
#                         "borderRadius": "5px",
#                         "border": "1px solid #ccc",
#                         "marginRight": "10px"
#                     }
#                 ),
#                 html.Button(
#                     "Get Code",
#                     id="get-code-btn",
#                     n_clicks=0,
#                     style={
#                         "padding": "10px 20px",
#                         "backgroundColor": "#083e7f",
#                         "color": "white",
#                         "border": "none",
#                         "borderRadius": "5px",
#                         "cursor": "pointer",
#                         "fontWeight": "bold"
#                     }
#                 )
#             ]
#         ),

#         # Response Output Section
#         html.Div(
#             id="response-output",
#             style={
#                 "width": "80%",
#                 "margin": "0 auto",
#                 "textAlign": "left"
#             }
#         )
#     ]
# )

# # --------------------- #
# #   DASH CALLBACK: Handle User Input
# # --------------------- #

# @app.callback(
#     Output("response-output", "children"),
#     Input("get-code-btn", "n_clicks"),
#     State("query-input", "value"),
# )
# def update_output(n_clicks, query):
#     """
#     When user clicks 'Get Code', retrieve similar job descriptions,
#     pass them to GPT, extract the occupation code, and display it.
#     """
#     if n_clicks > 0 and query:
#         retrieved_rows = query_system(query, df)
#         response_text = generate_response(query, retrieved_rows)

#         # Extract ISCO Code and Occupation from GPT Response
#         match = re.search(r"The occupational code is:\s*([\w\d]+)\s+for\s+(.*)\.", response_text)

#         if match:
#             isco_code = match.group(1)
#             occupation = match.group(2)

#             # Display Result in a Styled Card
#             return html.Div(
#                 style={
#                     "backgroundColor": "#444654",
#                     "borderRadius": "8px",
#                     "padding": "15px",
#                     "display": "flex",
#                     "alignItems": "center",
#                     "marginBottom": "10px"
#                 },
#                 children=[
#                     html.Div(
#                         isco_code,
#                         style={
#                             "fontWeight": "bold",
#                             "fontSize": "1.2em",
#                             "color": "#fff",
#                             "marginRight": "20px"
#                         }
#                     ),
#                     html.Div(
#                         occupation,
#                         style={
#                             "color": "#fff",
#                             "fontSize": "1em"
#                         }
#                     )
#                 ]
#             )
#         else:
#             # Display GPT's raw response if format is unexpected
#             return html.Div(
#                 response_text,
#                 style={
#                     "backgroundColor": "#444654",
#                     "color": "#fff",
#                     "borderRadius": "12px",
#                     "padding": "15px",
#                     "marginBottom": "10px"
#                 }
#             )

#     # Default empty response before any query
#     return html.Div(" ", style={"color": "#fff"})

# # --------------------- #
# #   RUN DASH SERVER
# # --------------------- #

# if __name__ == "__main__":
#     app.run_server(debug=True)


# # %%



