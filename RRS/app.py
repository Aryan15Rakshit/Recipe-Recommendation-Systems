from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

app = Flask(__name__)

UNSPLASH_ACCESS_KEY = "FVXZ0qKL1bTc83nUpGbYDZ-Q8jeRMBybKMzL0csMNec"

def get_food_image(query):
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": query,
        "client_id": UNSPLASH_ACCESS_KEY,
        "per_page": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data['results']:
        return data['results'][0]['urls']['regular']
    else:
        return "https://via.placeholder.com/300"



# Absolute path example
data = pd.read_csv("recipe_fixed.csv")

data['ingredients_list'] = data['ingredients_list'].str.lower()

data['image_url'] = "https://via.placeholder.com/300"


# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

# Combine Features
X_combined = np.hstack([
    X_numerical * 0.20,
    X_ingredients.toarray() * 0.80
])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(input_features):
    # scale numerical input
    input_features_scaled = scaler.transform([input_features[:7]])

    # transform ingredients
    input_ingredients_transformed = vectorizer.transform([input_features[7]])

    # combine features
    input_combined = np.hstack([
        input_features_scaled * 0.20,
        input_ingredients_transformed.toarray() * 0.80
    ])

    # get nearest neighbors
    distances, indices = knn.kneighbors(input_combined)

    # get recommendations
    recommendations = data.iloc[indices[0]]

    # select required columns
    recommendations = recommendations[['recipe_name', 'ingredients_list', 'image_url']].head(5)

    recommendations['image_url'] = recommendations['recipe_name'].apply(
        lambda x: get_food_image(x + " recipe food")
    )

    # force working image


    return recommendations


# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients'].lower()
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'),truncate = truncate)
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)