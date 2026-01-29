"""
This is the main script for Part 4 of the assignment:
- Loads the dataset and dictionary
- Initializes the local LLM
- Converts dataset rows to documents and builds vectorstore
- Runs the interactive query loop
"""
# Import necessary libraries
import pandas as pd
from langchain_ollama import OllamaLLM
from _part_4_3_FUNCTIONS import create_documents_from_df, create_vectorstore, create_vectorstore_and_save, load_vectorstore, query_loop
from _part_4_4_ADV_FUNCTIONS import query_loop_ADV

# Define whether to use advanced features or basic features
USE_ADVANCED = True     # or False for basic features

# (a) Basic Setup: Loading Data and Initializing the Model
# ------------------------------------------------------------------------
df = pd.read_csv("data_Part_4/patients.csv")      
dictionary = pd.read_csv("data_Part_4/index.csv")

print("Dataset loaded successfully.")
print(df.head())
print(dictionary.head())

# Initialize the model
llm = OllamaLLM(model="gemma:2b")
print("Ollama LLM initialized successfully.")
# ----------------------------------------------
# Example of a simple query to test the setup
#user_question = "Which features are most likely to influence survival?"

#prompt = f"""
#You are a medical data assistant. The dataset has columns: {', '.join(df.columns)}.
#Answer this question based on the dataset: {user_question}
#"""
# Send prompt to the model using the OllamaLLM instance
#response = llm.invoke(prompt)                            # .invoke() method is used to send prompts and receive responses
#print("LLM Answer:")
#print(response)
# ------------------------------------------------------------------------

# (b) RAG-style Retrieval
# ------------------------------------------------------------------------

# Convert dataset rows to documents
print("Converting dataset to documents...")
documents = create_documents_from_df(df)            

# Create FAISS vectorstore from documents
print("Creating FAISS Vectorstore...")
vectorstore = create_vectorstore(documents)
print("FAISS Vectorstore created!")

# ALTERNATIVE: Create and save vectorstore to disk (to avoid recomputation in future runs)
#print("Creating FAISS Vectorstore and saving to disk...")
#vectorstore = create_vectorstore_and_save(documents, "faiss_vectorstore")
#print("FAISS Vectorstore created and saved!")

# ALTERNATIVE: Load vectorstore from disk
#print("FAISS Vectorstore loaded from disk.")
#vectorstore = load_vectorstore("faiss_vectorstore")     # Load vectorstore from disk
#print("FAISS Vectorstore loaded!")
# ------------------------------------------------------------------------

# Query Loop to handle user questions 

if USE_ADVANCED:
    print("\nStarting Advanced Query Loop with Context and Advanced Numeric Analysis...\n")
# (i) Alternative more Advanced Query Features with Contextual Retrieval and History and advanced numeric analysis
# ------------------------------------------------------------------------
    # (with (f) Retrieval Summarization Function and (g)Advanced Numeric analysis tool inside (h)Advanced Agentic workflow and tool selection function)
    query_loop_ADV(vectorstore, llm, df, max_history=3)                    # max_history limits the number of previous interactions to consider
# ------------------------------------------------------------------------

else:
    print("\nStarting Basic Query Loop...\n")
# (e) Basic query loop to interact with the user
    # (with (c)Numerical analysis tool inside (d)Agentic workflow and tool selection function)
#------------------------------------------------------------------------
    query_loop(vectorstore, llm, df)
# ------------------------------------------------------------------------