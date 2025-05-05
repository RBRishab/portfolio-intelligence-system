from phi.tools import Toolkit
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
import time

class SentimentAnalysisToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="sentiment_analysis_toolkit")
        self.register(self.run_sentiment_analysis_for_stock)

        self.MODEL_NAME = "yiyanghkust/finbert-tone"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.labels = ['positive', 'negative', 'neutral']

    def fetch_headlines(self, stock: str, max_results=25, timeframe='d') -> list[str]:
        """Fetch headlines related to the stock using DuckDuckGo."""
        query = f"{stock} stock news"
        headlines = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='wt-wt', safesearch='Off', timelimit=timeframe):
                if r.get("title"):
                    headlines.append(r["title"])
                if len(headlines) >= max_results:
                    break
                time.sleep(0.1)  # Respect rate limits
        return list(set(headlines))

    def analyze_sentiment(self, headlines: list[str]) -> pd.DataFrame:
        """Use FinBERT to analyze sentiment of headlines."""
        inputs = self.tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

        results = []
        for idx, scores in enumerate(probs):
            label = self.labels[np.argmax(scores)]
            results.append({
                "headline": headlines[idx],
                "positive": round(float(scores[0]), 3),
                "negative": round(float(scores[1]), 3),
                "neutral": round(float(scores[2]), 3),
                "sentiment": label
            })
        return pd.DataFrame(results)

    def aggregate_sentiment(self, df: pd.DataFrame) -> dict:
        """Aggregate positive, negative, and neutral sentiment across headlines."""
        if df.empty:
            return {"positive": 0, "negative": 0, "neutral": 0, "sentiment_index": 0}
        sentiment_index = df["positive"].mean() - df["negative"].mean()
        return {
            "positive": round(float(df["positive"].mean()), 3),
            "negative": round(float(df["negative"].mean()), 3),
            "neutral": round(float(df["neutral"].mean()), 3),
            "sentiment_index": round(float(sentiment_index), 3)
        }

    def run_sentiment_analysis_for_stock(self, stock: str, timeframe: str = 'd') -> str:
        """
        Analyze sentiment for a given stock and return a clean string summary.
        Timeframe options: 'd' (day), 'w' (week), 'm' (month), 'y' (year)
        """
        headlines = self.fetch_headlines(stock, timeframe=timeframe)
        if not headlines:
            return f"❌ No headlines found for {stock.upper()}."

        df = self.analyze_sentiment(headlines)
        sentiment = self.aggregate_sentiment(df)

        return (
            f"📰 Sentiment analysis for {stock.upper()} (timeframe: {timeframe}):\n"
            f"- Positive: {sentiment['positive']}\n"
            f"- Negative: {sentiment['negative']}\n"
            f"- Neutral: {sentiment['neutral']}\n"
            f"- Sentiment Index (pos - neg): {sentiment['sentiment_index']}"
        )
