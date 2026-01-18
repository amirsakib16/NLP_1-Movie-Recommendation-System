from flask import Flask, request, jsonify, send_from_directory
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__, static_folder='.')

# Load data and models
df = joblib.load("df.joblib")
vectorizer = joblib.load("vectorizer.joblib")
vector_matrix = load_npz("vector_matrix.npz")

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "total_movies": len(df)
    })

@app.route("/movies", methods=["GET"])
def get_all_movies():
    """Get all movie titles"""
    movies = df["title"].tolist()
    return jsonify({"movies": movies})

@app.route("/recommend", methods=["POST"])
def recommend():
    """Get movie recommendations"""
    data = request.get_json()
    movie = data.get("movie", "").strip()
    
    if not movie:
        return jsonify({"error": "Movie name is required"}), 400
    
    # Find the movie (case-insensitive)
    movie_lower = movie.lower()
    matches = df[df["title"].str.lower() == movie_lower]
    
    if matches.empty:
        return jsonify({"error": "Movie not found"}), 404
    
    idx = matches.index[0]
    matched_title = df.iloc[idx]["title"]
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(vector_matrix[idx], vector_matrix)[0]
    
    # Get top 9 similar movies (excluding the input movie itself)
    similar_indices = similarity_scores.argsort()[::-1][1:10]
    recommendations = [df.iloc[i]["title"] for i in similar_indices]
    
    return jsonify({
        "matched": matched_title,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)