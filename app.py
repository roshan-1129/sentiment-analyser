import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
##inspect sample review 
word_index=imdb.get_word_index()

##reverse word index 
reverse_word_index={value:key for key,value in word_index.items()}

#load the model 
model=tf.keras.models.load_model('simplernn_imdb.h5')

#helper function 
#2 helper function to preprocess user input
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#preprocess user input
def preprocess_review(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]#2 is for unknown words
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#streamlit app
import streamlit as st
st.title("Sentiment Analyser using SimpleRNN")
user_review=st.text_area("Enter your movie review here:")
if st.button("Predict Sentiment"):
      preprocessed_input=preprocess_review(user_review)

      prediction=model.predict(preprocessed_input)
      sentiment='positive' if prediction[0][0]>=0.5 else 'negative'

      # Display the result
      st.write(f'Sentiment: {sentiment}')
      st.write(f'Score: {prediction[0][0]}')
else:
    st.write("Please enter a review and click the button to predict sentiment.")