# `LLMS-Project`

## Code state

> [!WARNING]
> Please be aware that for development purposes, these experiments use
> experimental Large Language Models (LLM's) not intended for production. They
> can present inaccurate information, hallucinated statements and offensive
> text by random chance or through malevolent prompts.

- **Under development** / **Experimental**
- **Depends on external API's**
   
## Prerequisites

## 1) Introduction

## 2) Installation
      
      
      You may want to use a virtual environment.
      Use terminal for this
      ```
      python -m venv NISR_LLM

      or 

      python3.11 -m venv NISR_LLM (if wanting specific python version)

      NISR_LLM\Scripts\activate
      ```

      ```
      python -m pip install --upgrade pip
      pip install -r requirements.txt
      ```

      Set-up custom kernel from your virtual environment (jupyter notebook)
      ```
      ipython kernel install --name "NISR_LLM" --user
      ```

      ### Installing occ_coder
      Navigate to `occ_coder` folder in your command line interface(CLI)

      The codebase is meant to also run as a python library, in order to install this
      you will then need to run:

      ```
      pip install --upgrade pip
      pip install .
      ```

      Use `pip list` to make sure you can see the `occ_coder = 0.0.1`

## 3) Usage

# License