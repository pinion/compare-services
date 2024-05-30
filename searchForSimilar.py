import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(files):
    # Read files and extract text
    file_texts = []
    for file_path in files:
        with open(file_path, 'r') as file:
            file_text = file.read()
            file_texts.append(file_text)

    # Create tf-idf vectorizer
    vectorizer = TfidfVectorizer()

    # Compute tf-idf matrix
    tfidf_matrix = vectorizer.fit_transform(file_texts)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    print (similarity_matrix)
    mask = np.abs(similarity_matrix) >=1
    print (np.unique(similarity_matrix[~mask])[-1])

    # Create a DataFrame for the similarity matrix
    # file_names = [os.path.basename(file) for file in files]
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
    similarity_df = pd.DataFrame(similarity_matrix, columns=file_names, index=file_names)

    return similarity_df

def compute_similarity(dataset_df, files):
    # Read files and extract text
    file_texts = []
    for file_path in files:
        with open(file_path, 'r') as file:
            file_text = file.read()
            file_texts.append(file_text)
    
    # Create tf-idf vectorizer
    vectorizer = TfidfVectorizer()

    # Compute tf-idf matrix for the query and file texts
    tfidf_matrix = vectorizer.fit_transform(file_texts)

    # Calculate cosine similarity between the query and file texts
    similarity_scores = cosine_similarity(dataset_df, tfidf_matrix).flatten
    similarity_scores = similarity_scores.argsort()

    # Create a DataFrame for the similarity scores
    similarity_df = pd.DataFrame({'Files': file_names, 'Similarity': similarity_scores})

    return similarity_df


def plot_similarity(similarity_df):
    cut_off=0.8
    # mask = np.abs(similarity_df) < cut_off
    similarity_df = similarity_df[similarity_df.Similarity > cut_off]
    print (similarity_df)
    sns.set(font_scale=0.8)
    sns.barplot(x='Similarity', y='Files', data=similarity_df, color='skyblue')
    sns.set(rc={'figure.figsize': (10, 8)})
    plt.title('Document Similarity')
    plt.show()


if __name__ == '__main__':
    # Specify the query document path
    query_document = './apiservices.jq'
    # query_document = './apispecs/accounts.json'

    # Specify the directory containing the documents
    document_directory = './apispecs'

    # Read the query document
    with open(query_document, 'r') as file:
        query_text = file.read()

    # Get all document files in the directory
    document_files = [os.path.join(document_directory, file) for file in os.listdir(document_directory)
                      if os.path.isfile(os.path.join(document_directory, file))]

    # Read document texts
    file_texts = []
    file_names = []
    for file in document_files:
        with open(file, 'r') as f:
            file_text = f.read()
            file_texts.append(file_text)
            file_names.append(os.path.basename(file))

    # API Specifications in the Asset
    documentroot = './compare-services/apispecs/'
    datasetFiles = [documentroot + 'users.json',
             documentroot + 'accounts.json'
             ]

    # Calculate similarity matrix
    dataset_df = calculate_similarity(datasetFiles)
    
    
    # similarity_df = compute_similarity(dataset_df, document_files)

    # Plot similarity
    plot_similarity(similarity_df)
