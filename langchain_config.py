# ===============================
# langchain_config.py - Direct API Keys
# ===============================

from gnews import GNews
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI  # For OpenAI usage if needed

# -----------------------------
# 🔑 Direct API Keys
# -----------------------------
OPENAI_API_KEY = "sk-proj-OPENAI_API_KEY"
NEWS_API_KEY = "NEWS_API_KEY"

# Initialize OpenAI LLM via LangChain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

# -----------------------------
# 📜 Prompt Template for Summarization
# -----------------------------
template = """
You are an AI assistant helping an equity research analyst.
Summarize the following news article.

Query: {query}

Article:
{article}

Return summary in max 4 sentences and give a sentiment (Positive, Negative, Neutral).
"""
prompt = PromptTemplate(template=template, input_variables=["query", "article"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# -----------------------------
# 📰 Fetch News Articles
# -----------------------------
def fetch_news(query: str, num_articles: int = 5, country: str = "IN", language: str = "en"):
    """
    Fetches news using GNews API (or NewsAPI if needed in future).
    """
    google_news = GNews(language=language, country=country, max_results=num_articles)
    articles = google_news.get_news(query)
    return articles or []

# -----------------------------
# 🔗 Main Function for app.py
# -----------------------------
def get_ai_summary_for_query(query: str, num_articles: int = 5, country: str = "IN", language: str = "en"):
    """
    Fetch news, summarize per article, and return overall summary.
    """
    # 1️⃣ Fetch articles
    articles = fetch_news(query, num_articles=num_articles, country=country, language=language)
    if not articles:
        return [], [], "No articles found."

    # 2️⃣ Remove duplicates by title
    unique_articles = []
    seen_titles = set()
    for a in articles:
        title = a.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(a)

    # 3️⃣ Per-article summaries
    per_article_summaries = []
    for a in unique_articles:
        article_text = f"Title: {a.get('title','')}\nDescription: {a.get('description','')}"
        ai_summary = llm_chain.run({"query": query, "article": article_text})

        per_article_summaries.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "summary": ai_summary,
            "sentiment": (
                "Positive" if "positive" in ai_summary.lower()
                else "Negative" if "negative" in ai_summary.lower()
                else "Neutral"
            )
        })

    # 4️⃣ Overall summary
    all_text = " ".join([s["summary"] for s in per_article_summaries])
    overall_summary = llm_chain.run({"query": query, "article": all_text})

    return unique_articles, per_article_summaries, overall_summary
