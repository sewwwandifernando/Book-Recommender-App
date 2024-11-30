from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
import numpy as np

app = Flask(__name__)
CORS(app, support_credentials=True)  # Enable CORS for all routes

# If you want to restrict it to a specific domain (localhost:3000):
# CORS(app, origins=["http://localhost:3000"])

# Load the model and other artifacts
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
similarity30 = pickle.load(open('artifacts/similarity.pkl', 'rb'))
new_df30 = pickle.load(open('artifacts/new_df.pkl', 'rb'))

alpha = 0.6  # Weight for collaborative filtering model
beta = 0.4   # Weight for content-based filtering model

# Function to get hybrid recommendations with poster URLs
def get_hybrid_recommendations(book_name):
    # Get collaborative filtering recommendations
    collaborative_books, collaborative_posters = collaborative_recommendations(book_name)
    
    # Get content-based filtering recommendations
    content_books, content_posters = content_recommendations(book_name)
    
    # Remove books in collaborative from content recommendations to avoid duplicates
    content_books = [book for book in content_books if book not in collaborative_books]
    content_posters = [poster for book, poster in zip(content_books, content_posters) if book not in collaborative_books]
    
    # Calculate total number of recommendations and count for each model
    total_recommendations = len(collaborative_books) + len(content_books)
    collab_count = int(alpha * total_recommendations)
    content_count = int(beta * total_recommendations)
    
    # Trim lists based on calculated counts
    collaborative_trimmed_books = collaborative_books[:collab_count]
    collaborative_trimmed_posters = collaborative_posters[:collab_count]
    
    content_trimmed_books = content_books[:content_count]
    content_trimmed_posters = content_posters[:content_count]
    
    # Combine both trimmed lists for final recommendations
    final_books = collaborative_trimmed_books + content_trimmed_books
    final_posters = collaborative_trimmed_posters + content_trimmed_posters
    
    return final_books, final_posters

def fetch_poster_collaborative(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

def fetch_poster_content(isbn_list):
    poster_url = []
    
    for isbn in isbn_list:
        # Check if the ISBN exists in the final_rating DataFrame
        if isbn in final_rating['ISBN'].values:
            # Fetch the image_url for the corresponding ISBN
            url = final_rating[final_rating['ISBN'] == isbn]['image_url'].values[0]
            poster_url.append(url)
        else:
            # If ISBN not found, append a placeholder or a default image URL
            poster_url.append('https://via.placeholder.com/150')  # Placeholder image URL
    
    return poster_url

def collaborative_recommendations(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    
    poster_url = fetch_poster_collaborative(suggestion)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    
    return books_list, poster_url

def content_recommendations(book):
    try:
        books_list = []
        # Find the index of the book in the DataFrame
        index = new_df30[new_df30['title'] == book].index[0]
        
        # Compute distances (similarity) and sort them in descending order
        distances = sorted(list(enumerate(similarity30[index])), reverse=True, key=lambda x: x[1])
        
        # Get the titles of the top 5 most similar books (excluding the first one, which is the book itself)
        recommended_books = [new_df30.iloc[i[0]].title for i in distances[1:6]]

        isbn_list = []
        for title in recommended_books:
            isbn = new_df30[new_df30['title'] == title]['ISBN'].values[0]
            isbn_list.append(isbn)
        
        poster_url = fetch_poster_content(isbn_list)
        
        return recommended_books, poster_url  # Return the list of recommended books
    
    except IndexError:
        # If the book is not found in new_df, return an empty list
        print(IndexError)
        return [''],['']

# API endpoint for getting recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    book_name = data.get('book_name')
    if not book_name:
        return jsonify({'error': 'No book name provided'}), 400
    
    final_recommendations, final_posters = get_hybrid_recommendations(book_name)
    
    return jsonify({
        'recommended_books': final_recommendations,
        'poster_url': final_posters
    })

# API endpoint for getting the list of book names
@app.route('/book-names', methods=['GET'])
@cross_origin(origin='*')
def get_book_names():
    # Convert the Pandas Index object to a list
    book_names_list = book_names.tolist()
    return jsonify(book_names_list)

if __name__ == '__main__':
    print("Flask app is running on http://127.0.0.1:5000")
    app.run(debug=True)
