from phi.tools import Toolkit
import openai
import os
import pandas as pd
import json
from typing import List, Dict
from pydantic import BaseModel

# Load your OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Pydantic Input for Risk Metrics ---
class RiskMetricsInput(BaseModel):
    Ticker: str
    Sharpe_Ratio: float
    VaR_95: float
    Max_Drawdown: float

# --- Explainability Toolkit ---
class ExplainabilityToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="explainability_toolkit")
        self.register(self.explain_portfolio_allocation)
        self.register(self.explain_sentiment)
        self.register(self.explain_risk_metrics)
        self.register(self.explain_financial_concept)

    def _call_gpt(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional financial analyst. Explain results, metrics, and financial concepts clearly and practically for investors."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ GPT API error: {e}"

    def explain_portfolio_allocation(self, allocation_data: List[Dict], strategy: str = "mpt") -> str:
        if not allocation_data:
            return "❌ No allocation data provided."
        df = pd.DataFrame(allocation_data)
        prompt = f"Explain this {strategy.upper()} portfolio allocation in detail like a financial analyst:\n\n{df.to_markdown(index=False)}"
        return self._call_gpt(prompt)

    def explain_sentiment(self, sentiment_data: Dict) -> str:
        if not sentiment_data or "stock" not in sentiment_data:
            return "❌ No sentiment data provided."
        summary = json.dumps(sentiment_data, indent=2)
        prompt = f"Explain the sentiment analysis result below in a way a retail investor would understand:\n\n{summary}"
        return self._call_gpt(prompt)

    def explain_risk_metrics(self, risk_data: RiskMetricsInput) -> str:
        summary = json.dumps(risk_data.dict(), indent=2)
        prompt = f"Explain the risk profile of {risk_data.Ticker} using the following risk metrics:\n\n{summary}"
        return self._call_gpt(prompt)

    def explain_financial_concept(self, query: str) -> str:
        if not query or len(query.strip()) < 5:
            return "❌ Please enter a valid financial concept or question."
        prompt = f"Explain this financial concept or question like a knowledgeable analyst:\n\n{query.strip()}"
        return self._call_gpt(prompt)
