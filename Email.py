import streamlit as st
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
image_path="inno12.png"
st.image(image_path)
model = pickle.load(open("nb.pkl","rb"))
with open("bow.pkl","rb") as f:
      bow=pickle.load(f)
st.title("MAIL SPAM-HAM CLASSIFIER")
email = st.text_input("enter the Email:")
if email:
      Data = bow.transform([email]).toarray()
      Result = model.predict(Data)[0]
if st.button("predict"):
      st.write("this email is a:",Result)
      if Result == "spam":
            image1="spam2.jpg"
            st.image(image1)
      else:
            image2="ham.jpg"
           st.image(image2) 
