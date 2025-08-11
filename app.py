import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast
import os

app = Flask(__name__)

# --- Data Loading and Preprocessing ---
# This part of the code replicates the steps from your Jupyter notebook
# to process the movie data and prepare the similarity matrix.

# Define file paths (assuming credits.csv and movies.csv are in the same directory)
CREDITS_PATH = 'credits.csv'
MOVIES_PATH = 'movies.csv'

# Check if files exist
if not os.path.exists(CREDITS_PATH) or not os.path.exists(MOVIES_PATH):
    print(f"Error: '{CREDITS_PATH}' or '{MOVIES_PATH}' not found.")
    print("Please ensure both 'credits.csv' and 'movies.csv' are in the same directory as app.py.")
    exit()

try:
    # Load datasets
    credits_df = pd.read_csv(CREDITS_PATH)
    movies_df = pd.read_csv(MOVIES_PATH)

    # Merge dataframes
    movies_df = movies_df.merge(credits_df, on="title")

    # Select relevant columns
    movies_df = movies_df[['movie_id','title','overview','genres','keywords','cast','crew']]

    # Drop rows with any missing values
    movies_df.dropna(inplace=True)

    # Function to convert stringified list of dictionaries to a list of names
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    # Function to extract top 3 cast members
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L

    # Function to fetch director's name
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L

    # Apply conversion functions
    movies_df['genres'] = movies_df['genres'].apply(convert)
    movies_df['keywords'] = movies_df['keywords'].apply(convert)
    movies_df['cast'] = movies_df['cast'].apply(convert3)
    movies_df['crew'] = movies_df['crew'].apply(fetch_director)

    # Convert overview to a list of words
    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

    # Remove spaces from names for consistent tagging
    movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ","") for i in x])

    # Create 'tags' column by concatenating relevant features
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew']

    # Create a new dataframe with only necessary columns
    new_df = movies_df[['movie_id','title','tags']]

    # Convert 'tags' list to string
    new_df['tags'] = new_df['tags'].apply(lambda x:' '.join(x))

    # Convert tags to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

    # Initialize Porter Stemmer for text stemming
    ps = PorterStemmer()

    # Stemming function
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)

    # Apply stemming to tags
    new_df['tags'] = new_df['tags'].apply(stem)

    # Vectorize text using CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)

    print("Data loading and preprocessing complete. Model ready!")

except Exception as e:
    print(f"An error occurred during data loading or preprocessing: {e}")
    print("Please check your CSV files and their content.")
    exit()

# --- Recommendation Function ---
def recommend(movie_title):
    """
    Recommends 5 similar movies based on the input movie title.
    """
    if movie_title not in new_df['title'].values:
        return [] # Movie not found

    movie_index = new_df[new_df['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    # Get top 5 most similar movies (excluding itself)
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(new_df.iloc[i[0]].title)
    return recommended_movies

# --- Flask Routes ---
@app.route('/')
def home():
    """
    Renders the main home page for the movie recommender.
    """
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Handles the POST request for movie recommendations.
    Expects a JSON payload with 'movie_title'.
    Returns JSON with recommended movies.
    """
    data = request.get_json()
    movie_title = data.get('movie_title')

    if not movie_title:
        return jsonify({'error': 'Movie title is required'}), 400

    recommendations = recommend(movie_title)
    if not recommendations:
        return jsonify({'message': 'Movie not found or no recommendations available.'}), 404

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True) # debug=True is good for development, set to False in production
