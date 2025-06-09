# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_and_train_model():
    # Load the dataset
    spam = pd.read_csv('mail_data.csv')

    # Encode labels
    encoder = LabelEncoder()
    Y = encoder.fit_transform(spam['Category'])  # ham = 0, spam = 1
    X = spam['Message']

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_tfidf, Y_train)

    # Evaluate accuracy
    train_accuracy = accuracy_score(Y_train, model.predict(X_train_tfidf))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test_tfidf))

    return model, vectorizer, train_accuracy, test_accuracy

# Load model and vectorizer
model, vectorizer, train_acc, test_acc = load_and_train_model()

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector", layout="centered")
st.title("📧 Spam Email Classifier")
st.markdown("Detect whether an email message is **Spam** or **Ham (Not Spam)** using a trained Machine Learning model.")

# Accuracy display
with st.expander("📊 Model Accuracy"):
    st.write(f"**Training Accuracy:** {train_acc:.2%}")
    st.write(f"**Test Accuracy:** {test_acc:.2%}")

# Input area
st.subheader("🔍 Enter the email text below:")
user_input = st.text_area("Your Email Text", height=200)

# Prediction
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter an email message to classify.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        if prediction == 1:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is a HAM (non-spam) email.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
