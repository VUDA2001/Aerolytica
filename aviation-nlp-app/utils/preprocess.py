import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

def preprocess(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalpha() and word not in stop_words]
import spacy
import streamlit as st

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")
