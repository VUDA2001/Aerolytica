import spacy
import streamlit as st
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_noun_phrases(text, top_n=30):
    doc = load_spacy()(text)
    noun_chunks = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    freq = Counter(noun_chunks).most_common(top_n)
    return [phrase for phrase, _ in freq]

def semantic_dendrogram(phrases):
    if not phrases:
        st.warning("No key phrases found.")
        return
    embeddings = load_embedding_model().encode(sorted(phrases))
    dist_matrix = cosine_distances(embeddings)
    linkage_matrix = linkage(dist_matrix, method="ward")
    fig, ax = plt.subplots(figsize=(10, 8))
    dendrogram(linkage_matrix, labels=sorted(phrases), orientation="right")
    st.pyplot(fig, use_container_width=True)

def summarize_dendrogram(phrases):
    categories = {
        "Pilot Error": {"pilot", "crew", "controller", "human"},
        "Mechanical Issue": {"engine", "gear", "brake", "fuel", "technical"},
        "Weather Condition": {"wind", "weather", "storm", "rain", "visibility"},
        "Procedural Error": {"checklist", "procedure", "protocol", "instruction"},
        "Communication": {"radio", "communication", "atc", "call"}
    }

    matched_categories = set()
    for phrase in phrases:
        for category, keywords in categories.items():
            if any(kw in phrase for kw in keywords):
                matched_categories.add(category)

    if matched_categories:
        joined = ", ".join(sorted(matched_categories))
        return f"**{len(phrases)} phrases clustered semantically.** Likely contributing themes include: {joined}."
    else:
        return f"**{len(phrases)} phrases clustered semantically.** No dominant cause category detected."

