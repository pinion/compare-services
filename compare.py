import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(files):
    # Read files and extract text
    file_texts = []

    with open("all_files.json", 'w') as write_file:
        write_file.write('[')
        for file_path in files:
            with open(file_path, 'r') as file:
                file_text = file.read()
                file_texts.append(file_text)
                write_file.write(file_text)
                write_file.write(',')
                file.close()
        write_file.write(']')
        write_file.close()
    
    
  

    # Create tf-idf vectorizer
    # vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer()  # Initialize CountVectorizer

        
    # Compute tf-idf matrix
    tfidf_matrix = vectorizer.fit_transform(file_texts) 
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Create a DataFrame for the similarity matrix
    # file_names = [os.path.basename(file) for file in files]
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
    similarity_df = pd.DataFrame(similarity_matrix, columns=file_names, index=file_names)

    return similarity_df


def plot_similarity(similarity_df):
    
    cut_off = 0.7
    corr = similarity_df.corr()
    # Create mask remove duplicates and comparisons to self
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Extend mask to all cells with value less than cut_off
    mask |= np.abs(corr) < cut_off
    # Set all masked cells to NaN
    corr = corr[~mask]
    
    # Drop all columns with all NaN
    corr = corr.dropna(axis=1, how='all')
    # Drop all rows with all NaN
    corr = corr.dropna(axis=0, how='all')
    
    plt.figure(figsize=(corr.shape))
    sns.set(font_scale=0.8)
    sns.heatmap(corr,
                vmin=cut_off,
                vmax=1, 
                square=True, 
                annot=True, 
                # robust=True,
                cmap="coolwarm",
                fmt='.1f',
                # yticklabels=corr.columns,
                # xticklabels=corr.columns, 
                cbar=True,
                # mask=mask
    )
    sns.set(rc={'figure.figsize': (corr.shape)})
    plt.title('API Specification Similarity')
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Specify the directory containing the files
    directory = './apispecs'
    print('pwd: ' + os.getcwd())

    
    # Set threshold for number of files to read
    n = 100
    # Get all files in the directory
    files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))][:n]

    # Get all files in the directory
    # files = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

    # Calculate similarity matrix
    similarity_df = calculate_similarity(files)

    # Plot similarity
    plot_similarity(similarity_df)
