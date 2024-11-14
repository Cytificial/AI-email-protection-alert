from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('email_threat_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define a route for classifying emails
@app.route('/classify', methods=['POST'])
def classify_email():
    # Get the email content from the POST request
    data = request.get_json()
    email_text = data.get('email', '')
    
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    # Convert the email text to TF-IDF features
    email_tfidf = vectorizer.transform([email_text])
    
    # Use the model to predict whether the email is spam or ham
    prediction = model.predict(email_tfidf)
    
    # Map the prediction to 'ham' or 'spam'
    label = 'spam' if prediction[0] == 1 else 'ham'
    
    # Return the result as a JSON response
    return jsonify({'prediction': label})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
