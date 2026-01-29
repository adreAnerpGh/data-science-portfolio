"""
Advanced functions for Part 4: Medical Data Analysis with LLMs
Includes:
- Summarization of retrieved rows - imported from _02_rag_functions.py
- LLM-powered numeric parsing and calculation - imported from _02_rag_functions.py
- Agentic workflow to choose the tool
- Advanced query loop with multi-turn context
"""


# Import necessary libraries
from langchain_ollama import OllamaLLM, OllamaEmbeddings                    # Ollama LLM and Embeddings for text generation and vectorization
from langchain_community.vectorstores.faiss import FAISS, Document          # FAISS vectorstore for efficient similarity search; Document class to hold text data

import pandas as pd
import json                                                                 # For data manipulation and JSON parsing                                

from _02_rag_functions import compute_numeric_analysis, agentic_workflow                            # import functions from _part_4_3_FUNCTIONS.py 


# (f) Retrieval Summarization Function
# ------------------------------------------------------------------------
# Function to summarize retrieved rows using the LLM
# use inside agentic_workflow() to return concise summary of retrieved rows instead of full text
def summarize_retrieved_rows(retrieved_text, llm):
    """
    Ask the LLM to summarize retrieved rows into concise info.
    Args:
        retrieved_text: Text of retrieved dataset rows.
        llm: OllamaLLM instance for text generation.
    Returns:
        summary: Summarized text from the LLM.
    """
    prompt = f"Summarize the following dataset rows in simple language:\n{retrieved_text}"
    summary = llm.invoke(prompt)
    return summary
# ------------------------------------------------------------------------


# (g) Numeric Analysis Tool using a mix of LLM parsing and direct DataFrame operations
# ------------------------------------------------------------------------
# LLM-powered numeric parsing + calculation
def compute_numeric_analysis_llm(df, question, llm):
    """
    Uses the LLM to parse the user's question and determine the numeric analysis to perform on the DataFrame.
    Then executes the analysis and returns the result.

    NOTE: If the LLM model is small (e.g., gemma:2b), it will likely fail to correctly parse the question
    into a JSON for numeric operations. Using a larger model (e.g., gemma:7b) improves reliability.

    Args:
        df (pd.DataFrame): The dataset.
        question (str): The user's question.
        llm (OllamaLLM): The LLM instance for parsing.
    Returns:
        result (str): The result of the numeric analysis.
    """
    # Create a prompt to ask the LLM how to analyze the data   
    prompt = f"""
    You are a Python data assistant. 
    Given the following question about this dataset: {question},
    Output a JSON with three fields:
    - "operation": "mean", "count", "correlation", "filter" taken from the question
    - "columns": list of columns involved (e.g., ["Age"] or ["Age","BloodPressure"]) from the dataset (LLM should choose relevant columns based on the question)
    - "value": numeric value if applicable (e.g., 60) taken from the question; null if not applicable.
    Respond ONLY in JSON format.                                       # JSON format makes it easier to parse the response programmatically
    """
    llm_output = llm.invoke(prompt)
#    print("DEBUG LLM OUTPUT:", llm_output)                # Debug: print the LLM output to verify the reasoning and JSON format

    try:                                                                    # Parse the LLM output as JSON to extract operation details
        parsed = json.loads(llm_output)
    except:                                                                 # If parsing fails due to invalid format, return an error message
        return "Sorry, I could not parse the numeric question."
    
    operation = parsed.get("operation")                                     # Extract operation type from parsed JSON (mean, count, correlation, filter)
    cols = parsed.get("columns", [])                                        # Extract list of columns involved from parsed JSON
    value = parsed.get("value", None)                                       # Extract numeric value if applicable from parsed JSON

    # Handle cases where multiple columns or operations are provided
    if len(cols) > 1 or isinstance(operation, list):                        # check whether both multiple columns and multiple operations are provided                   
        note = "Only the first operation was executed. Please ask one numeric operation at a time."   # using note instead of print to avoid spamming the output inside this function execution
    else:
        note = ""    

    # Execute the requested operation on the DataFrame
    if operation == "mean" and len(cols) == 1 and value is None:            # mean operation can be performed only on a single column; value (e.g., 60) must be provided for count operation
        col = cols[0]                                                       # get the column name (only the first element of the list because mean is for a single column); if multiple columns are provided, ignore the rest
        result_value = df[col].mean()
        return f"The mean of {col} is {result_value:.2f}.\n{note}"
    
    elif operation == "count" and len(cols) == 1 and value is not None:
        col = cols[0]
        count_value = df[df[col] == value].shape[0]
        return f"There are {count_value} rows where {col} is {value}.\n{note}"
    
    elif operation == "correlation" and len(cols) == 2 and value is None:
        col1, col2 = cols
        corr_value = df[col1].corr(df[col2])
        return f"The correlation between {col1} and {col2} is {corr_value:.2f}.\n{note}"
    
    elif operation == "filter" and len(cols) == 1 and value is not None:
        col = cols[0]
        filtered_rows = df[df[col] == value]
        return f"There are {filtered_rows.shape[0]} rows where {col} is {value}.\n{note}"
    
    else:
        return "Sorry, I could not perform the requested numeric analysis due to missing or invalid parameters."

# ------------------------------------------------------------------------

# (h) Agentic workflow: decide which tool to use based on the question
# ------------------------------------------------------------------------
def agentic_workflow_ADV(question, df, vectorstore, llm, history=None):                     # history parameter to support multi-turn context
    """
    Decide whether to use numeric analysis or RAG retrieval based on the user's question.
    Supports multi-turn context via `history`.
    Args:
        question: User's question as a string.
        df: DataFrame containing patient data.
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
        history: List of previous question/answer turns to include in context (optional)
    Returns:
        tool_used: The tool that was used ("numeric" or "rag").
        answer: The answer generated by the selected tool.
    """
    # Decide which tool to use
        # keywords indicating numeric operations to avoid false negative (not detecting numeric questions)
    numeric_keywords = ["mean", "average", "median", "avg", "min", "max", "count", "sum", "correlation", "std", "standard deviation"]

    # NUMERIC ANALYSIS PATH
    # ------------------------------------------------------------------------
    if any(k in question.lower() for k in numeric_keywords):
#        numeric_response = compute_numeric_analysis_llm(df, question, llm)             # not used currently due to reliability issues with Ollama gemma:2b
        numeric_response = compute_numeric_analysis(df, question)                       # use direct DataFrame operations for numeric analysis

        if "Sorry" not in numeric_response:     # if numeric analysis was successful (no error message), use numeric tool
            return "numeric", numeric_response
        
        tool_used = "rag"                       # return the tool=rag if numeric keywords were detected but the analysis failed
        
    else:
        return "rag"                            # return the tool=rag if numeric keywords were not detected
    # --------------------------------------------------------

    # RAG RETRIEVAL PATH
    # ------------------------------------------------------------------------
    results = vectorstore.similarity_search(question, k=3)  # get top 3 relevant documents from vectorstore
    retrieved_text = []

    for k in range(len(results)):
        retrieved_text.append("\n--- Retrieved Row ---\n")
        retrieved_text.append(results[k].page_content)
    retrieved_text = "\n".join(retrieved_text)  # combine retrieved texts into a single string

    summary = summarize_retrieved_rows(retrieved_text, llm)  # summarize retrieved rows using the LLM

    # Include history if available
    context_text = ""
    if history:
        context_text = "\n\nPrevious Conversation:\n" + "\n".join(history)

    final_prompt = f"""
    User Question:
    {question}

    {context_text}

    Relevant Dataset Rows:
    {summary}

    Answer using only this information.
    """
    llm_response = llm.invoke(final_prompt)
    if not llm_response:                                            # fallback in case LLM fails
        llm_response = "Sorry, I could not generate an answer."
# ------------------------------------------------------------------------
    # If numeric failed, still return RAG as answer
    if tool_used == "rag" and any(k in question.lower() for k in numeric_keywords) and "Sorry" not in numeric_response:
        
        return "numeric", numeric_response                          # If numeric actually succeeded, return numeric instead
    return tool_used, llm_response
# ------------------------------------------------------------------------


# (i) Advanced Query Features
# ------------------------------------------------------------------------
# Function for interactive query loop with multi-turn context (advanced alternative version to query_loop)
def query_loop_ADV(vectorstore, llm, df, max_history=3):               # 3 previous turns kept in context by default; more would increase prompt size and cost
    """
    Interactive query loop with multi-turn context for advanced queries.
    Uses agentic workflow to decide between numeric analysis and RAG retrieval.
    Args:
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
        df: DataFrame containing patient data.
        max_history: Maximum number of previous turns to keep in context.
    Returns:
        None                                                    # no return value; runs an interactive loop until user exits
    """
    print("RAG system with context ready. Type 'exit' to stop.")
    conversation_history = []                           # list to store previous turns in the conversation for context
    
    while True:
        user_q = input("Ask a question: ")
        if user_q.lower() == "exit":
            break
        
        tool_used, answer = agentic_workflow_ADV(user_q, df, vectorstore, llm, history=conversation_history)
        
        # Store conversation history (keep last max_history turns)
        conversation_history.append(f"User: {user_q}\nAnswer: {answer}")
        if len(conversation_history) > max_history:                             # limit history size by removing oldest turns if exceeding 3 previous turns
            conversation_history = conversation_history[-max_history:]          # keep only the last max_history turns; [-max_history:] gets the last max_history (3) elements from the list
        
        # Create context from conversation history for next turn, displaying the context to the user when showing the answer
        context = "\n\n".join(conversation_history)                     # combine history into a single string with double newlines between turns
        print(f"\n[{tool_used.upper()} Tool Result]:\n{answer}\n")      # .upper() to show tool used in uppercase
        print(f"\n[Conversation Context]:\n{context}\n")
# ------------------------------------------------------------------------


# https://docs.streamlit.io/get-started/fundamentals/main-concepts#understanding-streamlits-execution-model%20
# Note: In Streamlit, the entire script runs from top to bottom each time there is an interaction.
# Loops like while True should be avoided as they can freeze or block the app.

# Streamlit reruns the script from the top on every user action and normal variables like conversation_history reset each time.
# History will never grow beyond 1 turn unless we store it in st.session_state, which persists across reruns.

# (l) Basic and Advanced Query Loop to handle user questions in Streamlit app
# ------------------------------------------------------------------------
def query_loop_streamlit(vectorstore, llm, df, user_question):
    """
    Handle a single user question.
    Uses agentic workflow to decide between numeric analysis and RAG retrieval.
    Args:
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
        df: DataFrame containing patient data.
        user_question: The user's question as a string.
    Returns:
        tool_used: The tool that was used ("numeric" or "rag").
        answer: The answer generated by the selected tool.
    """
    tool, answer = agentic_workflow(user_question, df, vectorstore, llm)
    return tool, answer 
# ------------------------------------------------------------------------
def agentic_workflow_ADV_streamlit(user_question, df, vectorstore, llm, history=None):
    """
    Specialized agentic workflow for Advanced Streamlit.
    Decides whether to use numeric analysis (from dropdowns) or RAG retrieval.
    Preserves conversation history across interactions.
    
    Args:
        user_question: The user's question (free-text or numeric pseudo-question from dropdowns)
        df: DataFrame containing patient data
        vectorstore: FAISS vectorstore for retrieval
        llm: OllamaLLM instance for text generation
        history: List of previous conversation turns (optional)
    
    Returns:
        tool_used: The tool that was used ("numeric" or "rag")
        answer: The answer generated by the selected tool
    """
    # --------------------------------------------------------
    # Detect if numeric question based on keyword (for free-text numeric questions)
    numeric_keywords = ["mean", "average", "median", "avg", "min", "max", "count", "sum", "correlation", "std", "standard deviation"]
    is_numeric_question = False
    for keyword in numeric_keywords:
        if keyword in user_question.lower():
            is_numeric_question = True
            break
    # --------------------------------------------------------

    # NUMERIC ANALYSIS PATH
    # --------------------------------------------------------
    if is_numeric_question:
        numeric_response = compute_numeric_analysis(df, user_question)

        if "Sorry" not in numeric_response:  # numeric analysis successful
            tool_used = "numeric"
            answer = numeric_response
            return tool_used, answer
        # else continue to RAG
    # --------------------------------------------------------

    # RAG RETRIEVAL PATH
    # --------------------------------------------------------
    results = vectorstore.similarity_search(user_question, k=3)  # retrieve top 3 rows
    retrieved_text = []
    for k in range(len(results)):
        retrieved_text.append("\n--- Retrieved Row ---\n")
        retrieved_text.append(results[k].page_content)
    retrieved_text_combined = "\n".join(retrieved_text)

    # summarize retrieved rows using the LLM
    summary = summarize_retrieved_rows(retrieved_text_combined, llm)

    # include history in prompt if available
    context_text = ""
    if history is not None:
        if len(history) > 0:
            context_text = "\n\nPrevious Conversation:\n"
            for turn in history:
                context_text += str(turn) + "\n"

    # build final prompt for LLM
    final_prompt = "User Question:\n" + str(user_question) + "\n\n" + str(context_text) + "\nRelevant Dataset Rows:\n" + str(summary) + "\n\nAnswer using only this information."

    # call the LLM
    llm_response = llm.invoke(final_prompt)

    if not llm_response:  # fallback if LLM fails
        llm_response = "Sorry, I could not generate an answer."

    tool_used = "rag"
    answer = llm_response
    return tool_used, answer
# ------------------------------------------------------------------------

# Advanced Query Loop for Streamlit with multi-turn context
def query_loop_ADV_streamlit(vectorstore, llm, df, user_question, history, max_history=3):
    """
    Handle a single user question for advanced queries in Streamlit.
    Preserves conversation history across interactions.
    Args:
        vectorstore: FAISS vectorstore for retrieval.
        llm: OllamaLLM instance for text generation.
        df: DataFrame containing patient data.
        user_question: The user's question.
        history: List of previous turns in the conversation.
        max_history: Maximum number of previous turns to keep.
    Returns:
        tool_used: The tool that was used ("numeric" or "rag").
        answer: The answer generated by the selected tool.
        history: Updated conversation history.
    """
    tool_answer = agentic_workflow_ADV_streamlit(user_question, df, vectorstore, llm, history=history)
    if tool_answer is None:                                                 # fallback if something goes wrong inside agentic_workflow_ADV
        tool, answer = "rag", "Sorry, no answer could be generated."        # default to RAG with a generic error message
    else:
        tool, answer = tool_answer                                              # unpack the returned tuple from agentic_workflow_ADV
    
    # Update conversation history
    history.append(f"User: {user_question}\nAnswer: {answer}")

    # Limit history size by removing oldest turns if exceeding max_history
    if len(history) > max_history:
        history = history[-max_history:]

    return tool, answer, history
# ------------------------------------------------------------------------