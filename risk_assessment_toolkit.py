from phi.tools import Toolkit
import yfinance as yf
import pandas as pd
import numpy as np

class RiskMetricsToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="risk_metrics_toolkit")
        self.register(self.calculate_risk_metrics)

    def calculate_risk_metrics(self, tickers: list[str]) -> str:
        """
        Calculate Sharpe ratio, Value at Risk (95%), and Max Drawdown for one or more stock tickers.
        Accepts a list of ticker symbols.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        results = []

        for ticker in tickers:
            try:
                data = yf.download(ticker, period="1y", auto_adjust=True, progress=False)

                # Handle possible MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    close = data['Close'][ticker]
                else:
                    close = data["Close"]

                prices = close.dropna()
                if prices.empty:
                    results.append(f"❌ {ticker.upper()}: No valid price data.")
                    continue

                returns = prices.pct_change().dropna()
                if returns.empty or returns.std() == 0:
                    results.append(f"❌ {ticker.upper()}: Not enough variability to calculate risk.")
                    continue

                sharpe = returns.mean() / returns.std()
                var_95 = np.percentile(returns, 5)
                cumulative = (1 + returns).cumprod()
                drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()
                max_drawdown = drawdown.max()

                results.append(
                    f"📈 {ticker.upper()}:\n"
                    f"• Sharpe Ratio: {round(sharpe, 3)}\n"
                    f"• VaR (95%): {round(var_95, 3)}\n"
                    f"• Max Drawdown: {round(max_drawdown, 3)}\n"
                )

            except Exception as e:
                results.append(f"⚠️ {ticker.upper()}: Error calculating risk metrics: {str(e)}")

        return "\n".join(results)
