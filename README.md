# PDF Chatbot

This repository contains the code for the PDF Chatbot project. The goal of this project is to create an interactive chatbot that allows users to upload multiple PDF documents and ask questions about their content. The chatbot uses LangChain, Retrieval-Augmented Generation (RAG), Ollama (a lightweight model), and Streamlit for the user interface.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)


## Introduction

The PDF Chatbot project simplifies the process of extracting and querying information from multiple PDF documents. It leverages state-of-the-art natural language processing models and provides a user-friendly interface to interact with the documents.

## Features

- **Multiple Document Upload**: Upload and process multiple PDF documents simultaneously.
- **Interactive Q&A**: Ask questions and receive answers based on the uploaded documents.
- **User-Friendly Interface**: Built with Streamlit for ease of use.
- **Lightweight Model**: Utilizes Ollama for efficient processing.
- **Enhanced Retrieval**: Uses Retrieval-Augmented Generation (RAG) to improve response accuracy.

## Technologies Used

- **LangChain**: Framework for building applications with language models.
- **RAG (Retrieval-Augmented Generation)**: Combines retrieval and generation for more accurate answers.
- **Ollama**: Lightweight language model optimized for performance.
- **Streamlit**: Framework for creating interactive web applications with Python.

## Setup Instructions

Follow these steps to set up the project on your local machine:


**1. Clone the Repository**
Begin by cloning the repository to your local machine:
```
https://github.com/langchain-tech/chatgpt-clone-ollama-streamlit.git
cd chatgpt-clone-ollama-streamlit
```

**2. Create a Virtual Environment**
It is recommended to create a virtual environment to manage dependencies:
```
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
Install the necessary packages listed in the requirements.txt file:
```
pip install -r requirements.txt
```


**4. Set Up Environment Variables**
Create a .env file in the root directory of your project and add the required environment variables. For example:
```
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
```


**5. Start the Application**

Run the application using Streamlit:
```
streamlit run app.py
```
## Usage

1. **Upload Documents**: Use the interface to upload multiple PDF documents.
2. **Ask Questions**: Enter your questions in the provided text box.
3. **Get Answers**: The chatbot processes the documents and provides relevant answers based on the content.


