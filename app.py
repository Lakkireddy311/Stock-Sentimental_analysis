from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
randomclassifier = joblib.load('random_classifier.pkl')  # replace with your actual model file
countvectorizer = joblib.load('countvectorizer.pkl')    # replace with your actual vectorizer file

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        transformed_text = countvectorizer.transform([text])
        prediction = randomclassifier.predict(transformed_text)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"

        return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
