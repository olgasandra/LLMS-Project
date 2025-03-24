# `NISR LLM-Project`

[![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#experimental)
[![Shared under the MIT License](https://img.shields.io/badge/license-MIT-green)](https://github.com/datasciencecampus/Statschat/blob/main/LICENSE)

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

Large Language Models (LLMs) have significantly advanced natural language processing, enabling automation in tasks such as occupational code classification. Traditional classification methods, including the **International Standard Classification of Occupations (ISCO)**, **International Standard Classification of Education (ISCED)**, and **International Standard Industrial Classification (ISIC)**, often rely on manual input, which can be inefficient and prone to errors.

This project leverages LLMs to automatically analyze job titles, descriptions, and educational qualifications, assigning accurate occupational codes. By improving efficiency and scalability, this approach enhances labor market analysis and supports data-driven policy-making.

Developed as part of NISR’s Big Data and Data Revolution department, this project contributes to innovations in data processing and classification, driving Rwanda’s labor market intelligence forward.


## 2) Installation
      
### Virtual environment

**Once in the project space (i.e. the base repository level) it is recommended you set-up a virtual environment. In the terminal run:**
```
python -m venv NISR_LLM
```
**or** 
```
python3.11 -m venv NISR_LLM (if wanting specific python version)
```
**Activate your virtual environment**
```
NISR_LLM\Scripts\activate
```
**Upgrade pip**
```
python -m pip install --upgrade pip
```

**The codebase is meant to also run as a python library so the occ_coder package needs installing. Run:**

```     
pip install .
```
**Use `pip list` to make sure you can see `occ_coder = 1.0.1` (below is an older version)**

![image](https://github.com/user-attachments/assets/b7539bf2-9f69-49f5-ac69-885b8bd505e6)

**You may need to set-up a custom kernel from your virtual environment (if using jupyter notebook)**
```
ipython kernel install --name "NISR_LLM" --user
```

## 3) Usage

# License
