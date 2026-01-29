# Using Spacy and other relevant text processing tools to perform a descriptive analysis of the dataset
# (a) identifying Named Entities within each document using spaCy
# (b) generating TF-IDF vectors for each document with respect to the corpus using TfidfVectorizer
# (c) reporting the key terms for 10 randomly documents from the dataset (synthetic_docs folder) using TF-IDF scores

import spacy
import os  # added in order to use os.listdir and path.join
import random # to slect 10 random samples


############### (a) Named entity recognition (NER)

# Ensure the folder exists to prevent errors if the folder is missing
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"The folder {folder_path} does not exist. Please run the data generation script first.")

# Define the folder path to the documents
folder_path = "synthetic_docs"

# Create a spaCy language object
nlp = spacy.load("en_core_web_lg") # Model changed from md to lg, increasing the model size to get more accurate results

# To store document the content and the names of the files
corpus = []
names = []

# List of files in the folder
files = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        files.append(os.path.join(folder_path, filename))  

# Open the output file once before looping to avoid reopening the file for each entity. 
# Open the output file in write mode overwriting it every time the code is run.
with open("named_entities.txt", "w", encoding="utf-8") as out:      # utf-8 to deal with possible not ASCII symbols in the articles  

    # Process each file in the folder
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Even using lg model, the model tend to interpret symbols like #, **, etc. as entity such money or organizations.
        text = text.replace("#", "").replace("*", "").strip() # cleaning content before sending text to nlp

        # Add the content of each file into a corpus for later TF-IDF analysis
        corpus.append(text)         # corpus is a list of strings, each string is the content of one document

        # Process the text with spaCy
        doc = nlp(text)             # text is processed and stored in doc object (text is tokenized, parsed and entities are recognized)

        # Write header for clarity (document name)
        out.write(f"--- {os.path.basename(file_path)} ---\n")        # https://www.geeksforgeeks.org/python/python-os-path-basename-method/
        
        # Store the filename without the folder path
        names.append(os.path.basename(file_path))

        # Loop through entities and write to file
        for ent in doc.ents:
            out.write(f"{ent.text} ({ent.label_})\n")   # ent.text gives the actual entity string 
                                                        # ent.label_ gives the entity type (PERSON, ORG, GPE, etc.)
    
        # Add a blank line between documents for readability
        out.write("\n")

print("Named entity extraction complete. Results saved to named_entities.txt")

#####################################################

############### (b) generating TF-IDF vectors

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer
# Create TF-IDF vectors without stop_words (common words like "the", "is", etc. that do not carry significant meaning in text analysis)   
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the selected documents into a TF-IDF matrix
    # X is a sparse matrix; most values are 0 (because each document only contains a small subset of all terms); the 0s are stored but not explecitly represented.
X = vectorizer.fit_transform(corpus)  # X contains the TF-IDF weight of term j in doc i. Each row corresponds to a doc; each column corresponds to a unique term.

# Get the feature names 
    # feature_names is an array of all unique terms from the corpus, where each term’s position corresponds to its column index in the TF-IDF matrix X
    # The order of the terms is determined by the vocabulary mapping of TfidfVectorizer (not necessarily alphabetical)
feature_names = vectorizer.get_feature_names_out()   

# Get matrix dimensions
    # X.shape is a tuple of dimensions (num_docs, num_terms)  
num_docs, num_terms = X.shape    

print("TF-IDF matrix created for all documents.") 
print(f"There are {num_terms:,} unique terms across {num_docs} documents in the corpus.")

#####################################################

############### (c) Reporting the key terms for 10 randomly selected documents

# --------------------------------------------------------------------------

# https://stackoverflow.com/questions/66091139/how-to-find-important-words-using-tfidfvectorizer

# Converts the sparse TF-IDF matrix X to a dense matrix
X_dense = X.todense()   #  X_dense is now a full matrix where all zeros are explicitly represented, not just stored sparsely.

# Create a DataFrame from the dense matrix
# Row indices correspond to documents positions (0 to num_docs-1)
# Column headers correspond to each unique term (feature_names)
# Cells contain TF-IDF scores for each term in each document
X_df = pd.DataFrame(X_dense, columns=feature_names)  # Makes it easier to sort values and select the top-n terms per document

# --------------------------------------------------------------------------

# Randomly select 10 documents from the corpus
    # Create a sequence of document indices from 0 to (num_docs - 1)   -   https://www.w3schools.com/python/ref_func_range.asp
indices = range(X.shape[0]) # X.shape[0] returns the total number of documents (50)
    # # Randomly select 10 unique document indices from the corpus    -   https://www.geeksforgeeks.org/python/python-random-sample-function/
rand_indices = random.sample(indices, 10)  # random.sample(sequence, k) returns k = 10 1 distinct random documents

top_n = 10  # number of key terms per document

# Save the results to a text file
with open("key_terms.txt", "w", encoding="utf-8") as out:

    # Loop over the 10 randomly selected document indices
    for idx in rand_indices:    #  https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
        
        # Select the row corresponding to the document
        row = X_df.iloc[idx]   # row is a Series containing the TF-IDF scores for the document at position idx.
        # row.index is is the column headers of X_df (terms from feature_names)
        # row.values are the TF-IDF scores for each term in this idx
        # row.name is the row header (doc idx)

        # Sort the TF-IDF values in descending order and take the top_n terms
        sorted_terms = row.sort_values(ascending=False)
        top_terms = sorted_terms.head(top_n)   

        # Get the document name
        doc_name = names[idx]

        # Print the results
        print(f"\nDocument: {doc_name}")
        print("Top terms and their TF-IDF scores:")
        
        # Write the results to the text file
        out.write(f"--- {doc_name} ---\n")

        # Loop over the top terms using pandas index
        for term in top_terms.index:  # .index contains the actual terms(strings) of the top_terms row
            score = top_terms[term]  # access TF-IDF score
            print(f"{term}: {score:.4f}")       # score to 4 decimal for readability
            out.write(f"{term}: {score:.4f}\n")  
        out.write("\n")  # blank line between documents


