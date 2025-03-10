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
**Use `pip list` to make sure you can see `occ_coder = 1.0.0`**

![image](https://github.com/user-attachments/assets/b7539bf2-9f69-49f5-ac69-885b8bd505e6)

**You may need to set-up a custom kernel from your virtual environment (if using jupyter notebook)**
```
ipython kernel install --name "NISR_LLM" --user
```

## 3) Usage

# License
