#The main application file for the Aviation Safety Report Analyzer

import streamlit as st
from config import OPENAI_KEY, OPENAI_MODEL_NAME
from utils.file_io import extract_text_from_pdf
from utils.preprocess import preprocess
from utils.visualization import plot_wordcloud, summarize_wordcloud, plot_bigrams, summarize_bigrams, plot_dispersion
from utils.topic_modeling import compute_coherence_values, summarize_lda
from utils.clustering import extract_noun_phrases, summarize_dendrogram, semantic_dendrogram
from utils.knowledge_graph import extract_entity_relations, plot_knowledge_graph, summarize_knowledge_graph
from utils.chatbot import summarize_report, ask_question  # if using a function for chat
from utils.preprocess import load_spacy

import openai
import plotly.express as px
from gensim import corpora

openai.api_key = OPENAI_KEY

st.title("✈️AeroLytica")
st.markdown("Analyze aviation incident reports using NLP, AI, and interactive visualizations.")

uploaded_file = st.file_uploader("📎 Upload PDF report", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and preprocessing..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        tokens = preprocess(raw_text)
        processed_docs = [tokens]

    with st.expander("☁️ Word Cloud", expanded=False):
        plot_wordcloud(" ".join(tokens))
        wordcloud_summary, top_keywords = summarize_wordcloud(tokens)
        st.markdown(wordcloud_summary)


    with st.expander("📊 Top Bigrams"):
        plot_bigrams(processed_docs)
        st.markdown(summarize_bigrams(processed_docs))

    with st.expander("📌 Lexical Dispersion", expanded=False):
        plot_dispersion(tokens, top_keywords)

    with st.expander("🧪 Topic Modeling"):
        start = st.number_input("Start Topics", 2, 20, 2)
        limit = st.number_input("Max Topics", start+1, 30, 8)
        step = st.number_input("Step", 1, 5, 1)

        if st.button("Evaluate Topic Models"):
            with st.spinner("Running LDA..."):
                dictionary = corpora.Dictionary(processed_docs)
                corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                scores, models_list = compute_coherence_values(dictionary, corpus, processed_docs, start, limit, step)

                best_i = max(range(len(scores)), key=lambda i: scores[i][1])
                best_model = models_list[best_i]
                best_num_topics, best_score = scores[best_i]

                st.success(f"Best Topics: {best_num_topics} | Coherence: {best_score:.4f}")
                st.markdown(summarize_lda(best_model, best_num_topics))

                fig = px.line(x=[n for n, _ in scores], y=[c for _, c in scores], markers=True)
                fig.update_layout(title="Topic Coherence Scores", xaxis_title="Num Topics", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("🧠 Semantic Clustering"):
        phrases = extract_noun_phrases(raw_text, top_n=30)
        semantic_dendrogram(phrases)
        st.markdown(summarize_dendrogram(phrases))

    with st.expander("✈️ Entity Knowledge Graph"):
        nlp = load_spacy()
        triples = extract_entity_relations(raw_text, nlp)
        if triples:
            plot_knowledge_graph(triples)
            st.markdown(summarize_knowledge_graph(triples))
        else:
            st.info("No strong entity relationships detected.")

    with st.expander("📝 Executive Summary"):
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_report(raw_text)
                st.markdown(summary)
                st.session_state["summary"] = summary

        if "summary" in st.session_state:
            st.download_button(
                "📥 Download Summary",
                st.session_state["summary"],
                file_name="aviation_report_summary.txt",
                mime="text/plain"
            )

    with st.expander("🤖 Ask a Question"):
        user_query = st.text_area("Your question:", height=100)
        if user_query:
            with st.spinner("Generating answer..."):
                wordcloud_summary, top_keywords = summarize_wordcloud(tokens)

                graph_summaries = "\n\n".join([
                    wordcloud_summary,
                    summarize_bigrams(processed_docs),
                    summarize_lda(best_model, best_num_topics) if 'best_model' in locals() else "",
                    summarize_dendrogram(phrases),
                    summarize_knowledge_graph(triples) if triples else ""
                ])

                answer = ask_question(raw_text, user_query, graph_summaries)
                st.markdown(f"**Answer:** {answer}")
