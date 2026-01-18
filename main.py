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

@app.route("/movies", methods=["GET"])
def get_all_movies():
    movies = df["title"].tolist()
    return jsonify({"movies": movies})

@app.route("/recommend", methods=["GET"])
def recommend():
    movie = request.args.get("movie", "").lower()
    try:
        idx = df[df["title"].str.lower() == movie].index[0]
    except IndexError:
        return jsonify({"error": "Movie not found"}), 404
    similarity_scores = list(enumerate(cosine_similarity(vector_matrix[idx], vector_matrix)[0]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:10]
    recommendations = [df.iloc[i[0]].title for i in sorted_scores]
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)