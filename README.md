# 💼 Portfolio Intelligence System

A modular, multi-agent AI system for risk-aware portfolio optimization using MPT, HRP, sentiment analysis, and explainable AI — built with Python, Phidata, and LLMs.

---

## 📌 Project Overview

This capstone project showcases a modular, multi-agent AI system for risk-aware portfolio construction using:

* 📈 Advanced optimization strategies (MPT & HRP)
* 🧠 Real-time sentiment analysis (FinBERT + OpenBB)
* 💬 GPT-powered natural language explanations
* 🌐 A lightweight web UI built with Phidata Playground

---

## 🚀 Features

* **Risk Assessment Agent** – Calculates Sharpe Ratio, Value at Risk (VaR), and Max Drawdown
* **Sentiment Analysis Agent** – Monitors news and social media using FinBERT + OpenBB SDK
* **Portfolio Optimization Agent** – Allocates assets using MPT or HRP strategies
* **Explainability Agent** – Explains portfolio decisions using GPT (Groq or OpenAI)
* **Playground UI** – Chat with your agents using a local web app

---

## 🧱 Architecture

| Agent                       | Function                                                       |
| --------------------------- | -------------------------------------------------------------- |
| `RiskMetricsToolkit`        | Computes Sharpe, VaR, Max Drawdown                             |
| `PortfolioOptimizerToolkit` | Optimizes portfolios using MPT or HRP                          |
| `SentimentAnalysisToolkit`  | Extracts sentiment signals from financial news and social data |
| `ExplainabilityToolkit`     | Generates plain-language reasoning for portfolio decisions     |

---

## 🛠 Tech Stack

* **Python 3.12**
* **PyPortfolioOpt** – MPT and risk modeling
* **Riskfolio-Lib** – Advanced risk-based portfolio models
* **FinBERT** – Financial sentiment model for NLP
* **OpenBB SDK** – Real-time financial data
* **Groq LLaMA3 / OpenAI GPT-4** – Natural language reasoning
* **Phidata SDK** – Agent orchestration + Playground UI

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/capstone-financial-agent.git
cd capstone-financial-agent
```

---

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4. Add Your API Keys to a `.env` File

Create a file called `.env` in the root directory and add your keys:

```env
# .env

OPENAI_API_KEY=sk-...
GROQ_API_KEY=sk-...
PHI_API_KEY=pk-...
```

✅ These keys are used to power:

* **GPT-4** (OpenAI)
* **LLaMA3** (Groq)
* **Phidata Playground UI**

---

### 5. Run the Playground

```bash
python playground.py
```

Then open your browser and go to:

```
http://localhost:7777
```

✅ You'll see the Phidata Playground interface where you can chat with all your AI agents.

---

## 💬 Example Prompts

Try these in the Playground:

* “Optimize a portfolio with AAPL, TSLA, and NVDA using MPT for \$10,000 over 6 months with sentiment.”
* “What are the risk metrics for MSFT?”
* “Explain why GOOGL has a high VaR.”
* “Analyze sentiment for TSLA this week.”
* “Compare HRP vs MPT for NVDA and AAPL.”

---

## 👥 Contributors

| Name                    | Role                                                             |
| ----------------------- | ---------------------------------------------------------------- |
| **Rishab Reddy Bandi**  | Portfolio modeling, Explainability Agent, Phidata UI integration |
| **Deepthi Bhimanapati** | Risk & Sentiment Agents, OpenBB integration, testing             |

---

## 📚 References

* Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance
* Lopez de Prado, M. (2020). *Advances in Financial Machine Learning*
* Araci, D. (2019). *FinBERT: Financial Sentiment Analysis*
* PyPortfolioOpt – [https://github.com/robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt)
* Riskfolio-Lib – [https://riskfolio-lib.readthedocs.io](https://riskfolio-lib.readthedocs.io)
* OpenBB SDK – [https://openbb.co/sdk](https://openbb.co/sdk)

---

## ✅ License

This project is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**.  
➡️ You can share and build on this work for any purpose.  
➡️ **You must provide credit to the original author: Rishab Reddy Bandi**  
📖 [Read the license terms](https://creativecommons.org/licenses/by/4.0/)

