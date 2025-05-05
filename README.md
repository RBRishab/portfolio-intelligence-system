# portfolio-intelligence-system
A modular, multi-agent AI system for risk-aware portfolio optimization using MPT, HRP, sentiment analysis, and explainable AI — built with Python, Phidata, and LLMs.
# 💼 Multi-Agent AI-Powered Risk-Aware Portfolio Optimization

A capstone project showcasing a modular, multi-agent AI system for risk-aware portfolio construction using advanced optimization strategies (MPT & HRP), real-time sentiment analysis, and GPT-powered explainability.

---

## 🚀 Features

- 📊 **Risk Assessment Agent** — Computes Sharpe Ratio, Value at Risk (VaR), and Max Drawdown
- 🧠 **Sentiment Analysis Agent** — Uses FinBERT & OpenBB to analyze market news & social sentiment
- 📈 **Portfolio Optimization Agent** — Builds optimal portfolios using MPT & HRP based on user preferences
- 💬 **Explainability Agent** — Uses GPT (Groq or OpenAI) to explain portfolio decisions in plain language
- 💡 **Playground UI** — Built with [Phidata](https://phidata.com) for easy web-based interaction

---

## 🧱 Architecture Overview

The system is built on a modular **multi-agent architecture** where each agent performs a specific task:

| Agent               | Function                                                             |
|--------------------|----------------------------------------------------------------------|
| `RiskMetricsToolkit`       | Calculates Sharpe, VaR, and Drawdown                           |
| `PortfolioOptimizerToolkit` | Optimizes portfolios using MPT or HRP                          |
| `SentimentAnalysisToolkit`  | Analyzes financial news for bullish/bearish sentiment         |
| `ExplainabilityToolkit`     | Explains all results using GPT-powered natural language        |

---

## 📦 Tech Stack

- Python 3.12  
- [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)  
- Riskfolio-Lib  
- [FinBERT](https://github.com/ProsusAI/finBERT)  
- OpenBB SDK  
- Groq LLaMA3 or OpenAI GPT-4  
- Phidata SDK + Playground

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/capstone-financial-agent.git
cd capstone-financial-agent

2. Create and Activate a Virtual Environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Add Your API Keys to a .env File
Create a file called .env in the root directory:

env
Copy
Edit
# Example .env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=sk-...
PHI_API_KEY=pk-...
✅ These keys are used to power GPT-4 (OpenAI), LLaMA3 (Groq), and the Phidata UI.

▶️ Run the Playground
bash
Copy
Edit
python playground.py
