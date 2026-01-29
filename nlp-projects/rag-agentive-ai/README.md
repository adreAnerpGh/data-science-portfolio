# RAG & Agentive AI for Medical Data Analysis

This project implements a Retrieval-Augmented Generation (RAG) system combined with an agentive workflow to interact with a medical dataset using LLMs. Users can ask questions about the dataset, and the system decides whether to perform numeric analysis directly or retrieve relevant rows for context-aware LLM responses.

## Features

- Converts dataset rows into text documents for RAG retrieval.
- Builds or loads a FAISS vectorstore for fast similarity search.
- Supports numeric analysis (mean, sum, min, max, count, correlation) on the dataset.
- Advanced agentic workflow decides whether to use numeric computation or RAG retrieval.
- Multi-turn conversational context for advanced queries.
- Streamlit app integration for interactive use.

## Requirements

- Python 3.9+
- `pandas`
- `streamlit`
- `langchain_ollama`
- `faiss` / `langchain_community.vectorstores.faiss`

## Notes

- The embeddings model `gemma:2b` is used for both LLM inference and vector embeddings.
- The dataset (`patients.csv`) and feature dictionary (`index.csv`) are expected in `data_Part_4/`.
- The FAISS vectorstore should be created once and can be loaded from disk to avoid recomputation.
- **Pre-trained embeddings file `glove.6B.100d.txt` is not included in this repository**. Use the link provided in the code comments to download it if needed.

## Usage

1. Load the dataset and initialize the LLM.
2. Create or load the FAISS vectorstore.
3. Start the query loop:
   - Basic: free-text queries with RAG.
   - Advanced: numeric selection or free-text queries with multi-turn context.
4. Optionally, run via Streamlit for an interactive web interface.

```bash
streamlit run app_streamlit.py

