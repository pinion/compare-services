import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Data Preprocessing
def load_swagger_files(directory):
    swagger_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as file:
                swagger_files.append(json.load(file))
    return swagger_files

# Step 2: Feature Extraction
def extract_features(swagger_files):
    paths = []
    methods = []
    for swagger in swagger_files:
        paths.append(list(swagger.get("paths", {}).keys()))  # Extract endpoint paths
        methods.append(list(swagger.get("paths", {}).values()))  # Extract HTTP methods
    return paths, methods

# Step 3: Similarity Measurement
def calculate_similarity(features):
    vectorizer = CountVectorizer()  # Initialize CountVectorizer
    feature_matrix = vectorizer.fit_transform(features)  # Create feature matrix
    similarity_matrix = cosine_similarity(feature_matrix)  # Calculate cosine similarity
    return similarity_matrix

# Step 4: Data Loading
directory = "./apispecs"
swagger_files = load_swagger_files(directory)

# Step 5: Feature Extraction
paths, methods = extract_features(swagger_files)

# Step 6: Similarity Calculation
paths_similarity = calculate_similarity(paths)
methods_similarity = calculate_similarity(methods)

# Step 7: Example of Using Similarity Matrix
print("Similarity between Swagger files based on paths:")
print(paths_similarity)
print("Similarity between Swagger files based on HTTP methods:")
print(methods_similarity)
