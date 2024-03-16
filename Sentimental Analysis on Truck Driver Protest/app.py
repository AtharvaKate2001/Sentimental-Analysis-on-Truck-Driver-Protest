import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder="template", static_folder="staticFiles")

# Load the sentiment analysis model
sentiment_model = pickle.load(open('sentiment_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    text = request.form['text']

    # Make predictions using the sentiment analysis model
    sentiment = sentiment_model.predict([text])

    # Determine the result and sentiment text
    if sentiment == 'Positive':
        result_text = "Positive sentiment"
    else:
        result_text = "Negative sentiment"

    return render_template('result.html', prediction=result_text, sentiment=sentiment[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
