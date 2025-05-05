from phi.tools import Toolkit
import yfinance as yf
import pandas as pd
import numpy as np
import re
from pypfopt import EfficientFrontier, HRPOpt, expected_returns, risk_models
from sentiment_analysis_toolkit import SentimentAnalysisToolkit

class PortfolioOptimizerToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="portfolio_optimizer_toolkit")
        self.register(self.optimize_portfolio)
        self.sentiment_toolkit = SentimentAnalysisToolkit()

    def get_risk_free_rate(self) -> float:
        df = yf.download("^IRX", period="1d", interval="1d", auto_adjust=True)
        if df.empty or "Close" not in df.columns:
            raise ValueError("Could not retrieve risk-free rate.")
        return df["Close"].iloc[0].item() / 100.0

    def get_price_data(self, tickers: list[str], period: str) -> pd.DataFrame:
        data = yf.download(tickers, period=period, auto_adjust=True)["Close"]
        if isinstance(data, pd.Series):  # single stock
            data = data.to_frame()
        data.columns = data.columns.astype(str)
        return data.dropna()

    def get_sentiment_data(self, tickers: list[str], timeframe: str) -> pd.DataFrame:
        results = []
        for ticker in tickers:
            sentiment = self.sentiment_toolkit.run_sentiment_analysis_for_stock(ticker, timeframe)
            if isinstance(sentiment, str):  # error message
                continue
            results.append(sentiment)
        return pd.DataFrame(results)

    def adjust_returns_with_sentiment(self, prices: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.Series:
        mu = expected_returns.mean_historical_return(prices)
        sentiment_df = sentiment_df.set_index("stock")
        for stock in mu.index:
            if stock in sentiment_df.index:
                index = sentiment_df.loc[stock, "sentiment_index"]
                if pd.notna(index):
                    mu[stock] += 0.01 * index
        return mu

    def allocate_funds(self, weights: dict, prices: pd.DataFrame, investment_amount: float) -> pd.DataFrame:
        latest_prices = prices.iloc[-1]
        allocation = []
        for stock, weight in weights.items():
            if weight <= 0:
                continue
            allocated = investment_amount * weight
            quantity = int(allocated / latest_prices[stock])
            total_value = round(quantity * latest_prices[stock], 2)
            allocation.append({
                "Stock": stock,
                "Weight": round(weight, 4),
                "Price": round(latest_prices[stock], 2),
                "Quantity": quantity,
                "Total Value": total_value
            })
        return pd.DataFrame(allocation)

    def optimize_portfolio(
        self,
        tickers: list[str],
        investment_amount: float,
        period: str,
        timeframe: str,
        use_sentiment: bool,
        strategy: str
    ) -> str:
        try:
            # === Normalize strategy ===
            strategy = strategy.lower().strip()

            # === Normalize period ===
            period_input = str(period).strip().lower().replace(" ", "")
            if "month" in period_input or period_input.endswith("m"):
                digits = re.findall(r"\d+", period_input)
                period = f"{digits[0]}mo" if digits else "6mo"
            elif "year" in period_input or period_input.endswith("y"):
                digits = re.findall(r"\d+", period_input)
                period = f"{digits[0]}y" if digits else "1y"
            elif period_input in ["ytd", "max"]:
                period = period_input
            else:
                return f"❌ Unsupported period format: {period}"

            # === Normalize timeframe ===
            timeframe = str(timeframe).strip().lower()
            if "day" in timeframe or timeframe.startswith("d"):
                timeframe = "d"
            elif "week" in timeframe or timeframe.startswith("w"):
                timeframe = "w"
            elif "month" in timeframe or timeframe.startswith("m"):
                timeframe = "m"
            elif "year" in timeframe or timeframe.startswith("y"):
                timeframe = "y"
            else:
                timeframe = "m"  # default to monthly

            # === Validate Inputs ===
            if not tickers or not isinstance(tickers, list):
                return "❌ Please provide a list of stock tickers."
            if investment_amount <= 0:
                return "❌ Please provide a valid investment amount."
            if strategy not in ["mpt", "hrp"]:
                return "❌ Strategy must be either 'mpt' or 'hrp'."

            # === Fetch Data ===
            prices = self.get_price_data(tickers, period)
            if prices.empty:
                return f"❌ No valid price data for {tickers} during '{period}'."

            rf_rate = self.get_risk_free_rate()
            sentiment_df = self.get_sentiment_data(tickers, timeframe) if use_sentiment else None

            # === Strategy Logic ===
            if strategy == "mpt":
                mu = (
                    self.adjust_returns_with_sentiment(prices, sentiment_df)
                    if use_sentiment and sentiment_df is not None and not sentiment_df.empty
                    else expected_returns.mean_historical_return(prices)
                )
                S = risk_models.sample_cov(prices)
                ef = EfficientFrontier(mu, S)
                ef.add_constraint(lambda w: w <= 0.5)
                try:
                    ef.max_sharpe(risk_free_rate=rf_rate)
                except Exception:
                    ef.min_volatility()
                weights = ef.clean_weights()
                perf = ef.portfolio_performance(verbose=False)
            else:  # HRP
                rets = prices.pct_change().dropna()
                cov_matrix = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
                hrp = HRPOpt(rets, cov_matrix)
                hrp.optimize()
                weights = hrp.clean_weights()
                perf = hrp.portfolio_performance(verbose=False)

            allocation_df = self.allocate_funds(weights, prices, investment_amount)

            # === Format Output ===
            result = f"📈 Portfolio Optimization using {strategy.upper()}:\n"
            result += f"- Expected Return: {round(perf[0]*100, 2)}%\n"
            result += f"- Volatility: {round(perf[1]*100, 2)}%\n"
            result += f"- Sharpe Ratio: {round(perf[2], 2)}\n\n"
            result += "📊 Allocation:\n"
            for _, row in allocation_df.iterrows():
                result += f"{row['Stock']}: {row['Quantity']} shares at ${row['Price']} (Weight: {row['Weight']*100:.1f}%)\n"

            return result

        except Exception as e:
            return f"⚠️ Portfolio optimization failed: {str(e)}"
