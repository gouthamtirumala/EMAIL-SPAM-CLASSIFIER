import streamlit as st
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
st.image(r"C:\Users\HP\OneDrive\Desktop\inno12.png")
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
      if Result =="spam":
             st.image(r"C:\Users\HP\OneDrive\Desktop\spam2.jpg")
      else:
           st.image(r"C:\Users\HP\OneDrive\Desktop\ham.jpg") 
