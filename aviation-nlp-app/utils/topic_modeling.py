from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import plotly.express as px
import streamlit as st

def compute_coherence_values(dictionary, corpus, texts, start=2, limit=10, step=1):
    coherence_scores = []
    models_list = []
    for num_topics in range(start, limit + 1, step):
        model = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = coherencemodel.get_coherence()
        coherence_scores.append((num_topics, coherence))
        models_list.append(model)
    return coherence_scores, models_list

def summarize_lda(model, num_topics):
    lines = []
    for idx, topic in model.show_topics(num_topics=num_topics, formatted=False):
        terms = ", ".join([word for word, _ in topic])
        lines.append(f"- Topic {idx + 1}: {terms}")
    return "**Discovered Topics:**\n" + "\n".join(lines)

def plot_coherence_line(scores):
    x = [n for n, _ in scores]
    y = [c for _, c in scores]
    fig = px.line(x=x, y=y, markers=True)
    fig.update_layout(title="Topic Coherence Scores", xaxis_title="Num Topics", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

