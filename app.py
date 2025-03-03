from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import pandas as pd

# Load the trained KNN model and user-item matrix
knn_model = joblib.load(open('model_compressed2.pkl', 'rb'))  # Ensure you save your trained model as 'knn_model.pkl'
user_item_matrix = joblib.load(open('user_item_matrix2.pkl', 'rb'))  # Save your user-item matrix as 'user_item_matrix.pkl'

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_books():
    book_name = request.form.get('BookName')

    if book_name not in user_item_matrix.index:
        return render_template('index.html', result="Book not found in dataset.")

    # Find similar books using KNN
    book_idx = user_item_matrix.index.get_loc(book_name)
    distances, indices = knn_model.kneighbors(user_item_matrix.iloc[book_idx, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = [user_item_matrix.index[i] for i in indices.flatten()[1:]]  # Exclude the input book

    return render_template('index.html', result=recommended_books)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


