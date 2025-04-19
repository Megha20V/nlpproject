import modelbit
import streamlit as st
# data preprocessing
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocessor(data):
    # Convert to lowercase
    sentence = data.lower()
    
    # Remove punctuation
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
    # Remove numbers
    sentence = re.sub(r'\d+', '', sentence)
        
    # Remove stopwords and apply lemmatization
    # sentence = ' '.join([lemmatizer.lemmatize(word) for word in sentence.split() if word not in stop_words])
        
    # Append the processed sentence and its label
    processed_data = sentence    
    
    return processed_data

st.title("Sentence Analysis with Modelbit")

context = st.text_input("Enter the context sentence", "The food at the restaurant was highly recommended by many reviewers.")
final_sentence = st.text_input("Enter the final sentence", "Yeah, the food was so amazing that I had to spit it out.")

if st.button('Analyze Sentences'):
    # Process the final sentence
    processed_final_sentence = preprocessor(final_sentence)
    
    
    # Call modelbit inference
    result = modelbit.get_inference(
        region="us-east-2.aws",
        workspace="pugazhmukilan",
        deployment="analyze_sentence",
        data=[context, processed_final_sentence]
    )

    # Parse the result
    content_similarity = result["data"]["Content Similarity"]
    sentiment_shift = result["data"]["Sentiment Shift"]
    sarcasm_result = result["data"]["final_Sarcasm_Result"]

    # Display formatted result with better UX
    st.subheader("Inference Result:")

    # Content Similarity
    if content_similarity == "Yes":
        st.markdown("<b style='color:red;'>Content Similarity</b> ", unsafe_allow_html=True)
    else:
        st.markdown("<b style='color:green;'>No Content Similarity</b> ", unsafe_allow_html=True)

    # Sentiment Shift
    if sentiment_shift == "Yes":
        st.markdown("<b style='color:red;'>Sentiment Shift</b> ", unsafe_allow_html=True)
    else:
        st.markdown("<b style='color:green;'>No Sentiment Shift</b> ", unsafe_allow_html=True)

    if sarcasm_result == "Sarcastic":
        st.markdown("<b style='color:red;font-size: 50px'>Sarcasm Detected</b>", unsafe_allow_html=True)
    else:
        st.markdown("<b style='color:green;font-size: 50px'>Sarcasm Not Detected</b>", unsafe_allow_html=True)
