# Import necessary libraries
#--------------------------------------------------------
import streamlit as st                          # Streamlit for building web apps
import pandas as pd                
from langchain_ollama import OllamaLLM

# Import the functions
from _part_4_3_FUNCTIONS import create_documents_from_df, create_vectorstore, compute_numeric_analysis, load_vectorstore        # Basic functions
from _part_4_4_ADV_FUNCTIONS import query_loop_ADV_streamlit, query_loop_streamlit                    # Advanced functions adapted for Streamlit
# --------------------------------------------------------

# Streamlit App Title - https://docs.streamlit.io/develop/api-reference/text/st.title
st.title("Medical Data Assistant App")
# ------------------------------------------------

# Basic Setup: Loading Data and Initializing the Model
# --------------------------------------------------------
df = pd.read_csv("data_Part_4/patients.csv")      
dictionary = pd.read_csv("data_Part_4/index.csv")
# Use absolute paths for Streamlit app
    # Absolute paths were used to load the dataset and dictionary so that the Streamlit app can always find the files, 
        # even when it reruns the script after user interactions.
#path = r"C:\...\data_Part_4\patients.csv"
#dict_path = r"C:\Users\...\data_Part_4\index.csv"
#df = pd.read_csv(path)
#dictionary = pd.read_csv(dict_path)
# --------------------------------------------------------

# Display loaded data info - https://docs.streamlit.io/develop/api-reference/write-magic/st.write
# --------------------------------------------------------
st.write("Dataset loaded successfully.")
st.write(df.head())
st.write("Feature Dictionary:")
st.write(dictionary.head())
# --------------------------------------------------------

# Initialize the model
# --------------------------------------------------------
llm = OllamaLLM(model="gemma:2b")
# https://docs.streamlit.io/develop/api-reference/status/st.success
st.success("Ollama LLM initialized successfully.")      # .success() shows a success message in Streamlit; used instead of print() or write() for status updates
# --------------------------------------------------------

# RAG-style Retrieval - https://docs.streamlit.io/develop/api-reference/status/st.info
# --------------------------------------------------------
st.info("Converting dataset to documents...")
documents = create_documents_from_df(df)            # Convert dataset rows to documents
st.success("Documents created!")
#st.info("Creating FAISS Vectorstore...")
#vectorstore = create_vectorstore(documents)         # Create FAISS vectorstore from documents
#st.success("FAISS Vectorstore created!")
st.info("Loading FAISS Vectorstore from disk...")
vectorstore = load_vectorstore("faiss_vectorstore")     # Load vectorstore from disk
st.success("FAISS Vectorstore loaded!")
# --------------------------------------------------------

# Choice for Basic or Advanced Query Loop - https://docs.streamlit.io/develop/api-reference/widgets/st.radio
# --------------------------------------------------------
use_advanced = st.radio(
    "Select Query Mode:",
    ("Basic", "Advanced")
)
# --------------------------------------------------------

# Initialize conversation history in session_state (for Advanced mode)
# --------------------------------------------------------
# https://docs.streamlit.io/develop/api-reference/session-state/st.session_state
# session_state persists data across user interactions in Streamlit apps, allowing to maintain conversation history across multiple queries;
    # this overcomes the inherent nature of Streamlit apps where variables reset on each interaction
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []                                          # Initialize empty conversation history
# --------------------------------------------------------

# 1. ADVANCED METHOD
# --------------------------------------------------------
if use_advanced == "Advanced":                                          # If Advanced: ask whether user wants numeric-selection or free-text (RAG).
    # Ask the user which query type they want in Advanced mode
    query_type = st.radio(
        "Advanced mode — choose query type:",
        ("Numeric (select column & operation)", "Free-text (RAG)")
    )

    # 1.1 NUMERIC-SELECTION METHOD
    # ------------------------------------------------------------------------
    if query_type == "Numeric (select column & operation)":
        
        # https://docs.streamlit.io/develop/api-reference/write-magic/st.write
        st.write("Advanced Numeric Analysis Options")                           # Advanced numeric selection widgets (only show if Advanced & Numeric selected)

        # Numeric Selection Inputs for Advanced Analysis, to construct pseudo-questions that guide the numeric analysis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()          # select numeric columns from DataFrame and convert to list

        # https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox
        selected_col = st.selectbox("Select numeric column:", [""] + numeric_cols)                      # Dropdown for numeric columns
        selected_op = st.selectbox("Select operation:", ["", "mean", "sum", "min", "max", "std"])       # Dropdown for numeric operations

        # Trigger button so user confirms selection before execution - # https://docs.streamlit.io/develop/api-reference/widgets/st.button
        run_btn = st.button("Ask (run numeric operation)")

        if run_btn:
            # build pseudo question for internal function (compute_numeric_analysis expects a question-like string)
            question_to_use = f"{selected_op} of {selected_col.replace('_', ' ')}"          # replace underscores with spaces for matching 
        
            answer = compute_numeric_analysis(df, question_to_use)          # direct DataFrame operations for numeric analysis
            tool = "numeric"

            # Update history and show results
            st.session_state.conversation_history.append(f"User: {question_to_use}\nAnswer: {answer}")
            st.write(f"**Tool Used:** {tool}")
            st.write(f"**Answer:** {answer}")

            # Stop to persist UI and avoid re-run loops wiping selection - https://docs.streamlit.io/develop/api-reference/status/st.stop
            st.session_state["last_input"] = question_to_use
            st.stop()                                         # Stop execution to retain state after button press; prevents re-running and losing user selections 
    # ------------------------------------------------------------------------

    # 1.2 ADVANCED FREE-TEXT METHOD
    # ------------------------------------------------------------------------
    else:                                                               # If Basic: only free-text (RAG) is available.
        user_q = st.text_input(                                         # Advanced & Free-text; show a text input for RAG queries
            "Ask a free-text question for RAG (one at a time):",   
            value=st.session_state.get("last_input", "")                # Retain last input in session_state; .get() method retrieves value or default if not present
        )

        run_btn = st.button("Ask (RAG)")                                # Trigger button so user confirms input before execution        

        if run_btn and user_q:                                          # If button pressed and user question provided
            # Display conversation history
            st.write("Using Advanced Query Loop with Context and Advanced Numeric Analysis...")
            st.write("Conversation History")
            for turn in st.session_state.conversation_history:
                st.write(turn)
                st.write("---")

            # Use session_state to maintain conversation history
            tool, answer, st.session_state.conversation_history = query_loop_ADV_streamlit(
                vectorstore, llm, df,
                user_question=user_q,
                history=st.session_state.conversation_history,
                max_history=3
            )

            st.write(f"**Tool Used:** {tool}")
            st.write(f"**Answer:** {answer}")
            st.session_state["last_input"] = user_q
            st.stop()
# ------------------------------------------------------------------------

# 2. BASIC METHOD
# ------------------------------------------------------------------------
else:
    # Basic mode: only free-text (RAG). No advanced numeric-selection widgets shown.
    user_q = st.text_input(
        "Ask a question (one at a time, please):",
        value=st.session_state.get("last_input", "")
    )

    run_btn = st.button("Ask (Basic RAG)")

    if run_btn and user_q:
        st.write("Using Basic Query Loop...")
        tool, answer = query_loop_streamlit(vectorstore, llm, df, user_question=user_q)

        # Display the tool used and the answer (it helps to debug and understand the response)
        st.write(f"**Tool Used:** {tool}")
        st.write(f"**Answer:** {answer}")

        # Remember last input so user doesn’t need to retype
        st.session_state["last_input"] = user_q
        st.stop()
# ------------------------------------------------------------------------