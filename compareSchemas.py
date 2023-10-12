import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_openapi_specs(directory):
    spec_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    print('parsing files: ' + spec_files[0])
    specs = {}
    
    for filename in spec_files:
        print('Loading file: ' + filename)
        try:
            with open(os.path.join(directory, filename), 'r') as file:
                spec = json.load(file)
                # Extract and preprocess the schema section
                schema_section = json.dumps(spec.get("components", {}).get("schemas", {}), sort_keys=True)
                specs[filename] = schema_section
        except:
            print('Skipping file: ' + filename)
    return specs

def calculate_cosine_similarity(specs):
    # Create a TF-IDF vectorizer to convert schema sections into numerical vectors
    tfidf_vectorizer = TfidfVectorizer()
    schema_vectors = tfidf_vectorizer.fit_transform(specs.values())
    
    # Calculate cosine similarity between all pairs of specifications
    similarity_matrix = cosine_similarity(schema_vectors)
    
    return similarity_matrix

def generate_similarity_report(specs, similarity_matrix, threshold=0.7):
    report = {}
    num_specs = len(specs)
    
    for i, spec_file1 in enumerate(specs.keys()):
        report[spec_file1] = []
        for j, spec_file2 in enumerate(specs.keys()):
            if i != j and similarity_matrix[i][j] >= threshold:
                report[spec_file1].append(spec_file2)
    
    return report

def plot_similarity(similarity_matrix):
        # Create a DataFrame for the similarity matrix
        spec_files = list(specifications.keys())
        similarity_df = pd.DataFrame(similarity_matrix, columns=spec_files, index=spec_files)
        
        # Format plot
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
        
        # Set up the heatmap using seaborn
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
                cbar=True
        )
        
        sns.set(rc={'figure.figsize': (corr.shape)})
        plt.title("OpenAPI Specification Schema Similarity Heatmap")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

if __name__ == "__main__":
    # Directory containing the OpenAPI specification files
    spec_dir = "./apispecs"
    print('pwd: ' + os.getcwd())
    specifications = load_openapi_specs(spec_dir)
    
    if not specifications:
        print("No JSON files found in the directory.")
    else:
        
        similarity_matrix = calculate_cosine_similarity(specifications)
        similarity_threshold = 0.7  # Set your desired threshold
        
        similarity_report = generate_similarity_report(specifications, similarity_matrix, similarity_threshold)
        
        if not similarity_report:
            print("No similar schemas found.")
        else:
            print("Similarity Report:")
            for spec_file, similar_specs in similarity_report.items():
                if similar_specs:
                    print(f"Schema in '{spec_file}' is similar to schemas in: {', '.join(similar_specs)}")

        # Plot results
        plot_similarity(similarity_matrix)





