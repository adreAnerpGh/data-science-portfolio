# (a) Import necessary libraries
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ----------------------------------------------------------------------------


# (b) Load the copus (list of documents with text content as strings and names as list of strings)
# ----------------------------------------------------------------------------
# Keeping only the code that reads text files into memory as a list (`corpus`) and stores their filenames (`names`).

# To store document the content and the names of the files
corpus = []
names = []

# Ensure folder exists
os.makedirs("synthetic_docs", exist_ok=True)

# Define the folder path to the documents
folder_path = "synthetic_docs"      #  folder containing the text files created in Part 1

# List of files in the folder
files = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        files.append(os.path.join(folder_path, filename))  

# Process each file in the folder
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Even using lg model, the model tend to interpret symbols like #, **, etc. as entity such money or organizations.
    text = text.replace("#", "").replace("*", "").strip() # cleaning content before sending text to nlp

    # Add the content of each file into a corpus for later TF-IDF analysis
    corpus.append(text)    # corpus is a list of strings, each string is the content of one document

    # Store the filename without the folder path
    names.append(os.path.basename(file_path))
# ----------------------------------------------------------------------------


# (c) Create document representations (embeddings): CountVectorizer
# ----------------------------------------------------------------------------

# Create the Count Vectors that count word occurrences in each document (without common stop words)
count_vectorizer = CountVectorizer(stop_words='english')    
X_count = count_vectorizer.fit_transform(corpus)            # Fit and transform the selected documents into a Count matrix
print("CountVectorizer embeddings created.")
# ----------------------------------------------------------------------------


# (d) Create document representations (embeddings): TF-IDF Vectors
# ----------------------------------------------------------------------------

# Create TF-IDF Vectors that reflect the importance of words in each document relative to the entire corpus (without common stop words)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')        
X_tfidf = tfidf_vectorizer.fit_transform(corpus)                # sparse matrix of TF-IDF values
print("TF-IDF embeddings created.")
# ----------------------------------------------------------------------------


# (e) Create document representations (embeddings): GloVe Vectors
# ----------------------------------------------------------------------------

# Load GloVe embeddings
embeddings_index = {}                   # initialize an empty dictionary to hold the word embeddings from GloVe

# golbe.6B.100d.txt contains 400000 rows (words) and 101 columns (1 word + 100 vector components)

with open("glove.6B.100d.txt", encoding="utf-8") as glove:          # load the GloVe embeddings (100-dimensional) downloaded from https://downloads.cs.stanford.edu/unlp/data/glove.6B.zip
    for line in glove:                                              # each line contains a word followed by its 100-dimensional vector
        values = line.split()                                       # split the line into word and vector components
        word = values[0]                                            # the first element is the word
        coefs = np.asarray(values[1:], dtype='float32')             # the remaining elements are the vector components converted to float32
        embeddings_index[word] = coefs                              # store the word and its vector in the embeddings_index dictionary

# Create an empty document-term matrix for GloVe embeddings
document_embeddings = np.zeros((len(corpus), 100))          # 100-dimensional vectors for each document

# Compute the GloVe embedding for each document by averaging the embeddings of its words
for i in range(len(corpus)):                                # for each document
    tokens = corpus[i].split()                              # simple tokenization by splitting on whitespace
    
    valid_vectors = []                                      # to store valid GloVe vectors for tokens in each document as a list
    
    for token in tokens:                                            # for each token in the document 
        token_lower = token.lower()                                 # it takes the lowercase version of the token to match GloVe keys (glove.6B.100d.txt uses lowercase words)     
        if token_lower in embeddings_index:                         # check if the token of the document has a GloVe vector
            valid_vectors.append(embeddings_index[token_lower])     # add the GloVe vector to the list of valid vectors
                                                                    # embeddings_index is a dictionary mapping words to their GloVe vectors of 
    
    if valid_vectors:                                                   # if at least one token of the document has a related GloVe vector from the embeddings_index (400000 words)
        document_embeddings[i] = np.mean(valid_vectors, axis=0)         # compute the mean vector of all valid token vectors for the document
    else:                                                        # fallback for empty documents / no valid tokens
        document_embeddings[i] = np.zeros(100)                          # assign a zero vector if no valid tokens found

print("GloVe embeddings created.")


# (f) Similarity search functions: CountVectorizer
# ----------------------------------------------------------------------------

# Function to compute similarity for a new query using CountVectorizer
def query_count_vectorizer(query_text, vectorizer, X_count):
    """
    query: string, the query or new document
    vectorizer: CountVectorizer object used to create the originalcorpus embeddings
    X_count: sparse matrix, the CountVectorizer embeddings of the originalcorpus
    returns: similarity scores between the query and each document in the originalcorpus
    """
    query_vec = vectorizer.transform([query_text])                      # transform the query into Count Vector space instead of fitting again (to avoid changing the vocabulary of the original corpus)
    sim_scores = cosine_similarity(query_vec, X_count)[0]               # use cosine similarity to compare the query vector with the original corpus vectors
    return sim_scores                                                   # returns an array of similarity scores between the query and each document in the corpus
# ----------------------------------------------------------------------------


# (g) Similarity search functions: TF-IDF Vectors
# ----------------------------------------------------------------------------

# Function to compute similarity for a new query using TF-IDF Vectors

def query_tfidf_vectorizer(query_text, vectorizer, X_tfidf):
    """
    query: string, the query or new document
    vectorizer: TfidfVectorizer object used to create the originalcorpus embeddings
    X_tfidf: sparse matrix, the TF-IDF embeddings of the originalcorpus
    returns: similarity scores between the query and each document in the originalcorpus
    """
    query_vec = vectorizer.transform([query_text])                      # transform the query into TF-IDF space instead of fitting again (to avoid changing the vocabulary of the original corpus)
    sim_scores = cosine_similarity(query_vec, X_tfidf)[0]               # use cosine similarity to compare the query vector with the original corpus vectors
    return sim_scores                                                   # returns an array of similarity scores between the query and each document in the corpus
# ----------------------------------------------------------------------------


# (h) Similarity search functions: GloVe Vectors
# ----------------------------------------------------------------------------

# Function to compute similarity for a new query using GloVe Vectors
def query_glove_vectorizer(query_text, embeddings_index, document_embeddings):
    """
    query: string, the query or new document
    embeddings_index: dictionary mapping words to their GloVe vectors
    document_embeddings: numpy array, the GloVe embeddings of the originalcorpus
    returns: similarity scores between the query and each document in the originalcorpus
    """
    tokens = query_text.split()                                     # simple tokenization by splitting on whitespace
    valid_vectors = []                                              # to store valid GloVe vectors for tokens in the query as a list
    
    for token in tokens:                                                # for each token in the query
        token_lower = token.lower()                                     # convert to lowercase to match GloVe keys
        if token_lower in embeddings_index:                             # check if the token has a GloVe vector
            valid_vectors.append(embeddings_index[token_lower])         # add the GloVe vector to the list of valid vectors
    
    if valid_vectors:                                                   # if at least one token has a related GloVe vector
        query_vec = np.mean(valid_vectors, axis=0).reshape(1, -1)       # compute the mean vector of all valid token vectors for the query
    else:                                                               # fallback for empty queries / no valid tokens
        query_vec = np.zeros((1, 100))                                  # assign a zero vector if no valid tokens found
    
    sim_scores = cosine_similarity(query_vec, document_embeddings)[0]       # use cosine similarity to compare the query vector with the original corpus vectors
    return sim_scores                                                       # returns an array of similarity scores between the query and each document in the corpus
# -----------------------------------------------------------------------------

# (i) Read the queries from the files and perform similarity search
# ------------------------------------------------------------------------------

# Ensure folder exists
os.makedirs("queries", exist_ok=True)

# Define the folder path to the queries
queries_folder = "queries"      # folder containing the text files created in Part 3.1

# List of query files in the folder
query_files = []

for filename in os.listdir(queries_folder):
    if filename.endswith(".txt"):
        query_files.append(os.path.join(queries_folder, filename))
# ------------------------------------------------------------------------------

# (l) Perform similarity search for each query and display top 3 results for each method
# ------------------------------------------------------------------------------

# Define a helper function to get a snippet of text. It will be used to display a short excerpt from each document in the results, 
# to figure out if the document is actually relevant to the query.
def get_snippet(text):                                                      # get a snippet of the text for display (first sentence or first 120 characters)
    text = text.replace("\n", " ").strip()                                  # replace newlines with spaces and strip leading/trailing whitespace
    sentence_end = text.find(". ")                                          # find the end of the first sentence
    if sentence_end != -1:                                                  # if a period followed by a space is found
        return text[:sentence_end+1]                                        # return the first sentence including the period
    return text if len(text) < 120 else text[:120] + "..."                  # fallback: return up to 120 characters if no sentence end found

# Process each query file in the folder
for query_file in query_files:
    with open(query_file, "r", encoding="utf-8") as f:
        query_text = f.read().strip()                                       # read the query text and remove leading/trailing whitespace

    # Perform similarity search using CountVectorizer
    count_sim_scores = query_count_vectorizer(query_text, count_vectorizer, X_count)

    # Perform similarity search using TF-IDF Vectors
    tfidf_sim_scores = query_tfidf_vectorizer(query_text, tfidf_vectorizer, X_tfidf)

    # Perform similarity search using GloVe Vectors
    glove_sim_scores = query_glove_vectorizer(query_text, embeddings_index, document_embeddings)

    # Display top 3 results for each method
    print(f"\nQuery: {query_text}\n")

    # CountVectorizer top-3
    print("CountVectorizer top-3:")
    top3_count_indices = np.argsort(count_sim_scores)[-3:][::-1]            # argsort returns indices that would sort the array; [-3:][::-1] gets the top 3 in descending order
    for rank, idx in enumerate(top3_count_indices, start=1):                # enumerate to get rank starting from 1
        snippet = get_snippet(corpus[idx])                                  # get a short snippet of the document for display
        print(f"{rank}. Document: {names[idx]}, Similarity Score: {count_sim_scores[idx]:.4f}")
        print(f"Snippet: {snippet}")

    # TF-IDF top-3
    print("\nTF-IDF top-3:")
    top3_tfidf_indices = np.argsort(tfidf_sim_scores)[-3:][::-1]
    for rank, idx in enumerate(top3_tfidf_indices, start=1):
        snippet = get_snippet(corpus[idx])                                  # get a short snippet of the document for display
        print(f"{rank}. Document: {names[idx]}, Similarity Score: {tfidf_sim_scores[idx]:.4f}")
        print(f"Snippet: {snippet}")

    # GloVe top-3
    print("\nGloVe top-3:")
    top3_glove_indices = np.argsort(glove_sim_scores)[-3:][::-1]
    for rank, idx in enumerate(top3_glove_indices, start=1):
        snippet = get_snippet(corpus[idx])                                  # get a short snippet of the document for display
        print(f"{rank}. Document: {names[idx]}, Similarity Score: {glove_sim_scores[idx]:.4f}")
        print(f"Snippet: {snippet}")

    print("\n" + "-"*80)        # Separator between queries