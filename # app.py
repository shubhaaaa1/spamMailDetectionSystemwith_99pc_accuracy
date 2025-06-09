# app.py

import streamlit as st

# Streamlit UI
st.set_page_config(page_title="Spam Email Detector", layout="centered")

st.title("📧 Spam Email Classifier")
st.markdown("Detect whether an email message is **Spam** or **Ham (Not Spam)** using a trained Machine Learning model.")

st.subheader("🔍 Enter the email text below:")
input_text = st.text_area("Your Email Text", height=200)

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        transformed_input = vectorizer.transform([input_text])
        prediction = model.predict(transformed_input)[0]

        if prediction == 1:
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is a HAM (non-spam) email.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Scikit-Learn")
