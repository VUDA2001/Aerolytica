import openai

def summarize_report(text, model="gpt-3.5-turbo"):
    prompt = f"""
You are an aviation incident analyst. Summarize this aviation accident report in a structured format.

Instructions:
- Overview (aircraft, date, location)
- Chronology
- Contributing factors
- Outcome
- Final probable cause and Key insights

Report Text:
{text[:3500]}
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You analyze aviation safety reports and generate structured summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=700
    )
    return response["choices"][0]["message"]["content"]

def ask_question(report_text, user_query, graph_summaries, model="gpt-3.5-turbo"):
    prompt = f"""
You are a helpful aviation safety assistant. Use the following aviation report and summarized insights to answer the user's question.

Report:
{report_text[:2000]}

Visual Insights:
{graph_summaries}

Question:
{user_query}
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in aviation safety analysis. Use structured data and visual insights to answer questions about reports."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]
