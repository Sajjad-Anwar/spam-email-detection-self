import streamlit as st
import joblib

# Load the trained model and vectorizer from saved files
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Title and description of the app
st.title("📧 Spam Email Detection System")
st.write("Enter the content of the email below, and the model will predict if it's **Spam** or **Not Spam**.")

# Input text box for the user to enter email content
input_text = st.text_area("✉️ Email Content:")

# Predict button
if st.button("🚀 Predict"):
    # Check if the input text is not empty
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some email content to classify.")
    else:
        # Convert the input text to numerical features using the vectorizer
        input_tfidf = vectorizer.transform([input_text])

        # Use the model to predict if the email is spam or not
        prediction = model.predict(input_tfidf)[0]

        # Display the prediction result
        if prediction == 1:
            st.error("🚫 This is a **Spam Email**!")
        else:
            st.success("✅ This is **Not Spam**.")
