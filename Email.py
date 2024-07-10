import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Path to images (assuming they are in the same directory as the script)
image_path = "inno12.png"
image_spam = "spam2.jpg"
image_ham = "ham.jpg"

# Display an image
st.image(image_path)

# Load the trained model
try:
    with open("nb.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the CountVectorizer (bow) used during model training
try:
    with open("bow.pkl", "rb") as f:
        bow = pickle.load(f)
except Exception as e:
    st.error(f"Error loading CountVectorizer (bow): {e}")
    st.stop()

# Streamlit UI
st.title("MAIL SPAM-HAM CLASSIFIER")

email = st.text_input("Enter the Email:")

if st.button("Predict"):
    if email:
        # Transform the input email using the CountVectorizer (bow)
        email_vectorized = bow.transform([email]).toarray()
        
        # Make prediction
        prediction = model.predict(email_vectorized)[0]
        
        # Display the result
        st.write(f"This email is: {prediction}")
        
        # Display corresponding image based on prediction
        if prediction == "spam":
            st.image(image_spam)
        else:
            st.image(image_ham)
    else:
        st.error("Please enter an email to predict.")
