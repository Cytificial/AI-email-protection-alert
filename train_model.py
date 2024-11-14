import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import seaborn as sns  # After installing seaborn
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/SMSSpamCollection', encoding='latin-1', sep='\t', header=None, names=['label', 'message'])

# Data Preprocessing
# Convert labels to binary: 'spam' -> 1, 'ham' -> 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data into features and labels
X = df['message']
y = df['label']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, 'email_threat_model.pkl')  # Save model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save vectorizer

# Model evaluation
y_pred = model.predict(X_test_tfidf)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))
