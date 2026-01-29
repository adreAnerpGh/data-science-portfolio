"""
This file contains all reusable functions for Part 4 of the assignment:
- Converting dataset rows into documents
- Creating a FAISS vectorstore with embeddings
- Performing numeric analysis on the DataFrame
- Running an interactive query loop for RAG-style retrieval and LLM answering
"""

# Import necessary libraries
from langchain_ollama import OllamaLLM, OllamaEmbeddings                    # Ollama LLM and Embeddings for text generation and vectorization
from langchain_community.vectorstores.faiss import FAISS, Document          # FAISS vectorstore for efficient similarity search; Document class to hold text data

import pandas as pd

# (b) RAG-style Retrieval Functions
# ------------------------------------------------------------------------
# Function to convert each row of the dataset into a simple text document
def create_documents_from_df(df):
    """
    Convert each row of the dataframe into a text document.
    Args:
        df: DataFrame containing patient data.
    Returns:
        List of Document objects representing each patient.
    """
    documents = []                                       # initialize empty list to hold documents

    for i in range(len(df)):                             # iterate over each row in the dataframe (patients) ; len(df) gives number of rows
        row = df.iloc[i]                                 # get the i-th row as a Series object
        text_parts = []                                  # initialize list to hold text parts for this document

        for j in range(len(df.columns)):                 # iterate over each column index (features); len(df.columns) gives number of columns
            col_name = df.columns[j]
            value = row[col_name]
            text_parts.append(f"{col_name}: {value}")    # create text part for this feature and append to text_parts list

        full_text = "\n".join(text_parts)                # combine all text parts into a single string (text document); \n to separate features

        doc = Document(page_content=full_text, metadata={"row_index": i})     # create Document object combining the text string and row index of a patient
        documents.append(doc)                                                 
    return documents

# Function to create a FAISS vectorstore from the documents
    # the vectorstore allows efficient similarity search for RAG retrieval as it stores vector embeddings of the documents instead of raw text (faster search and lower memory usage)
def create_vectorstore(documents):
    """
    Converts documents into embeddings and builds a FAISS vectorstore.
    Returns the vectorstore object.
    Args:
        documents: List of Document objects.
    Returns:
        FAISS vectorstore built from the documents.
    """
    # initialize embeddings model to convert text documents into vector representations
    embeddings_model = OllamaEmbeddings(model="gemma:2b")
    # create FAISS vectorstore from documents and embeddings model                  
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    return vectorstore

def create_vectorstore_and_save(documents, filepath):
    """
    Converts documents into embeddings, builds a FAISS vectorstore, and saves it to disk.
    Returns the vectorstore object.
    Args:
        documents: List of Document objects.
        filepath: Path to save the FAISS vectorstore.
    Returns:
        FAISS vectorstore built from the documents.
    """
    # initialize embeddings model to convert text documents into vector representations
    embeddings_model = OllamaEmbeddings(model="gemma:2b")
    # create FAISS vectorstore from documents and embeddings model                  
    vectorstore = FAISS.from_documents(documents, embeddings_model)
    # save the vectorstore to disk
    vectorstore.save_local(filepath)
    return vectorstore

def load_vectorstore(filepath):
    """
    Loads a FAISS vectorstore from disk.
    Returns the vectorstore object.
    Args:
        filepath: Path to load the FAISS vectorstore from.
    Returns:
        FAISS vectorstore loaded from disk.
    """
    embeddings_model =OllamaEmbeddings(model="gemma:2b")        # initialize embeddings model
    vectorstore = FAISS.load_local(                             # load the FAISS vectorstore from disk using the embeddings model
    folder_path=filepath,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True)  
    return vectorstore
# ------------------------------------------------------------------------


# (c) Numeric Analysis Tool
# ------------------------------------------------------------------------
# Function to perform numeric analysis based on user question
def compute_numeric_analysis(df, question):
    """
    Analyze the user question to identify and perform numeric computations on the DataFrame.
    Supported computations: average, correlation, count.
    Returns the results as a string.
    Args:
        df: DataFrame containing patient data.
        question: User question as a string.
    Returns:
        Result of the numeric computation as a string.
    """
    q = question.lower()                                                # convert question to lowercase for easier matching
    result_text = "Sorry, I cannot compute this request."               # default response if no computation is performed

    # Try to match numeric operations
    for col in df.columns:
        col_lower = col.lower().replace("_", " ")                       # normalize column name for matching (lowercase, replace underscores with spaces)    
        if col_lower in q:                                              # approximate match for column names
            
            if "average" in q or "mean" in q:                           # Mean / Average
                value = df[col].mean()
                return f"The mean/average of {col} is {value:.2f}."
            
            if "sum" in q:                                              # Sum
                value = df[col].sum()
                return f"The sum of {col} is {value:.2f}."
            
            if "min" in q or "minimum" in q:                            # Min
                value = df[col].min()
                return f"The minimum of {col} is {value:.2f}."
            
            if "max" in q or "maximum" in q:                            # Max
                value = df[col].max()
                return f"The maximum of {col} is {value:.2f}."
            
            if "std" in q or "standard deviation" in q:                 # Standard Deviation
                value = df[col].std()
                return f"The standard deviation of {col} is {value:.2f}."
            
            if "count" in q:
                for val in df[col].unique():                            # Count specific value
                    if str(val).lower() in q:
                        count_val = df[df[col] == val].shape[0]
                        return f"There are {count_val} rows where {col} is {val}."

    # Correlation (pair of columns)
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                col1_lower = col1.lower().replace("_", " ")
                col2_lower = col2.lower().replace("_", " ")

                if "correlation" in q or "correlate" in q:                 
                    if col1_lower in q and col2_lower in q:
                        value = df[col1].corr(df[col2])                                         # .corr() method computes correlation between two columns
                        return f"The correlation between {col1} and {col2} is {value:.2f}."

    return result_text
# ------------------------------------------------------------------------

# (d) Agentic Workflow / Tool Selection
# ------------------------------------------------------------------------
# Function to decide which tool to use based on the question
def agentic_workflow(question, df, vectorstore, llm):
    """
    Decide whether to use the numeric analysis tool or RAG retrieval + LLM based on the question.
    Returns the tool used and the corresponding response.
    Args:
        question: User question as a string.
        df: DataFrame containing patient data.
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
    Returns:
        tool_used: String indicating which tool was used ("numeric" or "rag").
        response: The response generated by the selected tool.
    """
    print("\n[Agentic Workflow] Deciding which tool to use...\n")
    # decide which tool to use
    numeric_response = compute_numeric_analysis(df, question)   # use numeric analysis tool to check if numeric computation is needed

    if "Sorry" not in numeric_response:
        return "numeric", numeric_response
    else:
        # do RAG retrieval + LLM
        results = vectorstore.similarity_search(question, k=3)
        retrieved_text = []
        for k in range(len(results)):
            retrieved_text.append("\n--- Retrieved Row ---\n")
            retrieved_text.append(results[k].page_content)

        retrieved_text = "\n".join(retrieved_text)                  # combine retrieved texts into a single string
        
        final_prompt = f"""
        User Question:
        {question}

        Relevant Dataset Rows:
        {retrieved_text}

        Answer using only this information.
        """
        llm_response = llm.invoke(final_prompt)
        return "rag", llm_response
# ------------------------------------------------------------------------

# (e) Query Loop to handle user questions
# ------------------------------------------------------------------------
# Function for interactive query loop to handle user questions
def query_loop(vectorstore, llm, df):
    """
    Interactive query loop for user questions.
    Uses agentic workflow to decide between numeric analysis and RAG retrieval.
    Args:
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
        df: DataFrame containing patient data.
    Returns:
        None                                                    # no return value; runs an interactive loop until user exits
    """
    print("RAG system ready. Type 'exit' to stop.")
    while True:
        user_q = input("Ask a question: ")
        if user_q.lower() == "exit":
            break
        tool_used, answer = agentic_workflow(user_q, df, vectorstore, llm)      # decide which tool to use and get the answer
        print(f"\n[{tool_used.upper()} Tool Result]:\n{answer}")
# ------------------------------------------------------------------------