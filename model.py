import joblib

# Load the trained model
model = joblib.load('data/email_threat_model.pkl')

def predict_threat(body):
    """
    Given the body of the email, predict the threat level using the trained model.
    Returns "high" for spam and "low" for non-spam (ham).
    """
    prediction = model.predict([body])
    return "high" if prediction[0] == 1 else "low"
