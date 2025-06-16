import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk import bigrams

def plot_wordcloud(text):
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

def summarize_wordcloud(tokens, top_n=20):
    freq = Counter(tokens).most_common(top_n)
    summary = "**Top Terms Identified:**\n" + "\n".join([f"- `{word}`: {count}" for word, count in freq])
    top_keywords = [word for word, _ in freq]
    return summary, top_keywords


def plot_dispersion(tokens, keywords):
    word_positions = {word: [] for word in keywords}
    for i, token in enumerate(tokens):
        if token in word_positions:
            word_positions[token].append(i)
    if not any(word_positions.values()):
        st.info("No keywords from word cloud found in the text.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, word in enumerate(keywords):
        ax.plot(word_positions[word], [i] * len(word_positions[word]), '|', label=word)
    ax.set_yticks(range(len(keywords)))
    ax.set_yticklabels(keywords)
    ax.set_title("Lexical Dispersion Plot (Top Terms from Word Cloud)")
    st.pyplot(fig, use_container_width=True)


def plot_bigrams(docs):
    all_words = [word for doc in docs for word in doc if len(word) > 2]
    bigram_list = list(bigrams(all_words))
    bigram_freq = Counter(bigram_list).most_common(15)
    labels = [f"{w1} {w2}" for (w1, w2), _ in bigram_freq]
    counts = [count for _, count in bigram_freq]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels[::-1], counts[::-1])
    st.pyplot(fig, use_container_width=True)

def summarize_bigrams(docs, top_n=10):
    all_words = [word for doc in docs for word in doc if len(word) > 2]
    bigram_list = list(bigrams(all_words))
    freq = Counter(bigram_list).most_common(top_n)
    return "**Top Bigrams:**\n" + "\n".join([f"- `{w1} {w2}`: {count}" for (w1, w2), count in freq])

