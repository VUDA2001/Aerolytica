import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

def extract_entity_relations(text, nlp, top_n=30):
    doc = nlp(text)
    triples = []
    meaningless = {"he", "she", "it", "they", "this", "that", "these", "those", "someone", "something"}
    for sent in doc.sents:
        subj, verb, obj = '', '', ''
        for token in sent:
            if 'subj' in token.dep_:
                subj = token.text.lower()
            if token.pos_ == 'VERB':
                verb = token.lemma_
            if 'obj' in token.dep_:
                obj = token.text.lower()
        if subj and verb and obj and subj not in meaningless and obj not in meaningless:
            triples.append((subj, verb, obj))
    return list(set(triples))[:top_n]

def summarize_knowledge_graph(triples, top_n=5):
    return "**Top Relationships:**\n" + "\n".join([f"- `{s}` → `{v}` → `{o}`" for s, v, o in triples[:top_n]])

def plot_knowledge_graph(triples):
    G = nx.DiGraph()
    for e1, rel, e2 in triples:
        G.add_edge(e1, e2, label=rel)
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, edge_color='gray', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=8)
    st.pyplot(plt.gcf(), use_container_width=True)
    plt.clf()

