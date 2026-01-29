# RAG Agentive AI - Streamlit App

This folder contains a Streamlit-based app for **RAG (Retrieval-Augmented Generation) with agentive workflow**.  
The app allows you to query medical data using either numeric analysis or free-text retrieval with a local LLM.

## Requirements

- Python 3.11+  
- Streamlit (`pip install streamlit`)  
- LangChain Ollama (`pip install langchain-ollama`)  
- FAISS (`pip install faiss-cpu`)  
- Pandas (`pip install pandas`)  

Make sure the `faiss_vectorstore` and dataset files (`patients.csv`, `index.csv`) are present in the folder.

## Running the App

After cloning the repository:

```bash
# Navigate to the project root folder (replace <repo-folder> with your path)
cd <repo-folder>

# Run the Streamlit app
streamlit run rag-agentive-ai/app_streamlit.py
