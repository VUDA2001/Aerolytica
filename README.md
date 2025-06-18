![avigif](https://github.com/user-attachments/assets/b167e2f8-e33d-481e-bd5d-e4578cd6a193)
# ✈️ AeroLytica: Aviation NLP Report Analyzer

**AeroLytica** is a Streamlit-based NLP application that analyzes aviation accident and incident reports(you can download them from https://www.ntsb.gov/investigations/AccidentReports/Pages/Reports.aspx). It extracts critical insights such as frequent terms, key bigrams, topic distributions, and causal relationships — all interactively visualized. Users can upload a report and explore both structured analytics and natural language Q&A.

---

##  Features

- **PDF Upload & Parsing** – Upload NTSB or FAA incident reports
- **Word Cloud + Summary** – Highlight frequent keywords visually and textually
- **Top Bigrams** – Identify frequent co-occurring term pairs
- **Lexical Dispersion Plot** – Visualize keyword density across report structure
- **Topic Modeling** – Coherence-based LDA tuning to extract key themes
- **Semantic Clustering** – Group noun phrases using transformer embeddings
- **Knowledge Graph** – Visualize subject–verb–object relationships
- **Integrated GPT-3.5 Chatbot** – Ask plain-English questions about the report
- **Downloadable Summary Report** – Export top terms, topics, and graphs

---

## Tech Stack

- `Python 3.9+`
- `Streamlit` – UI/UX framework
- `NLTK`, `SpaCy`, `Gensim` – Core NLP processing
- `Matplotlib`, `WordCloud`, `NetworkX` – Visualizations
- `SentenceTransformers` – Semantic embeddings
- `OpenAI API` – GPT-based chatbot

## Future Enhancements

- Multi-report comparison and auto-clustering
- GPT-4 model integration
- FAA/NTSB API ingestion
- PDF export of summary insights

#### If you found this project useful or interesting, please consider ⭐ starring it on GitHub!
