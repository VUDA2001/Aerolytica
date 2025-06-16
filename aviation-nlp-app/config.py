# config.py

import streamlit as st

# Set Streamlit page config early
st.set_page_config(page_title="Aviation NLP", layout="wide")

# OpenAI Key
OPENAI_KEY = st.secrets["openai_key"]

# Target keywords for lexical dispersion
TARGET_KEYWORDS = ['accident', 'aircraft', 'pilot', 'flight', 'safety', 'faa', 'performance', 'report']

# Sentence transformer model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Spacy model
SPACY_MODEL_NAME = "en_core_web_sm"

# OpenAI model
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
