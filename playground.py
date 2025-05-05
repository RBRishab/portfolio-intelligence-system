import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app

# Load env
load_dotenv()
os.environ["PHIDATA_API_KEY"] = os.getenv("PHIDATA_API_KEY")

# === Import Your Custom Toolkits ===
from risk_assessment_toolkit import RiskMetricsToolkit
from sentiment_analysis_toolkit import SentimentAnalysisToolkit
from portfolio_optimizer_toolkit import PortfolioOptimizerToolkit
from explainability_toolkit import ExplainabilityToolkit

# === Multi-Agent Financial Analyst ===
aggregator_agent = Agent(
    name="Multi-Agent Financial Analyst",
    role="Answer all investment and financial questions using specialized toolkits.",
    model=Groq(id="llama3-70b-8192"),  # Or OpenAIChat(id="gpt-4")
    tools=[
        YFinanceTools(
            stock_price=True,
            stock_fundamentals=True,
            analyst_recommendations=True,
            company_news=True
        ),
        RiskMetricsToolkit(),
        SentimentAnalysisToolkit(),
        PortfolioOptimizerToolkit(),
        ExplainabilityToolkit()
    ],
    instructions=[
        "Use tables when showing multiple stocks.",
        "Use RiskMetricsToolkit for risk explanations, PortfolioOptimizerToolkit for MPT/HRP, and ExplainabilityToolkit for user reasoning.",
        "Always explain your reasoning clearly."
    ],
    show_tool_calls=True,
    markdown=True
)

# === Create Playground App ===
app = Playground(agents=[aggregator_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
