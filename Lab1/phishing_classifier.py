"""
AI-Powered Phishing Email Classifier Lab Exercise
--------------------------------------------------
This script demonstrates a complete pipeline to build a phishing email classifier.
It covers:
- Data creation (sample CSV generation)
- Text cleaning and preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction using TF-IDF
- Data splitting, model training (using Logistic Regression), and evaluation
- Prediction on a new email sample

This exercise is written for beginners and includes detailed explanations for every step.
"""

# ---------------------------
# 1. Import Required Libraries
# ---------------------------
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# 2. Download NLTK Resources
# ---------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ---------------------------
# 3. Data Acquisition: Create Sample Dataset
# ---------------------------
# For demonstration, we create a small sample dataset and save it as a CSV file.
data_samples = {
    'Email_Content': [
        "Congratulations! You have won a prize. Click here to claim your reward.",
        "Dear customer, your account has been updated successfully.",
        "Urgent: Update your payment information to avoid suspension.",
        "Meeting rescheduled to 3 PM tomorrow."
    ],
    'Label': [
        "phishing", "legitimate", "phishing", "legitimate"
    ]
}
# Create a DataFrame from the sample data
df_samples = pd.DataFrame(data_samples)
# Save the DataFrame as a CSV file (this simulates having an external dataset)
df_samples.to_csv("emails.csv", index=False)

# Load the dataset from the CSV file
data = pd.read_csv('emails.csv')
print("Initial Data:")
print(data.head())

# ---------------------------
# 4. Data Preprocessing
# ---------------------------

# 4.1 Define a function to clean text: remove unwanted characters and lowercase the text.
def clean_text(text):
    # Remove non-alphabetical characters and convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

# Apply the cleaning function to the email content
data['Cleaned_Email'] = data['Email_Content'].apply(clean_text)

# 4.2 Define a function to tokenize, remove stopwords, and perform lemmatization.
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the cleaned email text
data['Processed_Email'] = data['Cleaned_Email'].apply(preprocess_text)
print("\nProcessed Email Samples:")
print(data[['Email_Content', 'Processed_Email']].head())

# ---------------------------
# 5. Feature Extraction Using TF-IDF
# ---------------------------
tfidf_vectorizer = TfidfVectorizer()
# Transform the processed email text into TF-IDF features
X_features = tfidf_vectorizer.fit_transform(data['Processed_Email'])

# Convert labels into numerical values (phishing: 1, legitimate: 0)
y = data['Label'].apply(lambda x: 1 if x == 'phishing' else 0)

# ---------------------------
# 6. Splitting the Data
# ---------------------------
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.2, random_state=42
)
print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ---------------------------
# 7. Model Training
# ---------------------------
# Using Logistic Regression as the classifier
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# ---------------------------
# 8. Model Evaluation
# ---------------------------
# Predict the labels for the test set
y_pred = model_lr.predict(X_test)

# Evaluate the model's performance using accuracy, classification report, and confusion matrix.
print("\nModel Evaluation (Logistic Regression):")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 9. Prediction Function for New Emails
# ---------------------------
def predict_email(email_text, model, vectorizer):
    """
    Given an email text, preprocess it, convert to TF-IDF features,
    and predict whether it's a phishing or legitimate email.
    """
    cleaned = clean_text(email_text)
    processed = preprocess_text(cleaned)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)
    return "Phishing Email" if prediction[0] == 1 else "Legitimate Email"

# Test the prediction function with a new email example.
new_email = "Urgent! Your account is compromised. Click the link to secure your account immediately."
result = predict_email(new_email, model_lr, tfidf_vectorizer)
print("\nPrediction for New Email:")
print("Email:", new_email)
print("Result:", result)
