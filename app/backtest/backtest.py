import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.optimize import curve_fit

from app import slack
from app.utils import get_current_dir, write_to_log

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)

NUM_FORMAT = "{:.4f}"
CURRENT_DIR = get_current_dir()


class Backtest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.tax_rate = 0.001
        self.transaction_fee = 0.001

        for key, value in kwargs.items():
            setattr(self, key, value)

    def execute(self, df: pd.DataFrame) -> None:
        self.capital = self.initial_capital
        self.shares = 0

        self.df = df
        self.preprocess()
        for i in range(len(df)):
            self.trade(df.iloc[i])
        self.consolidate_results()
        self.plot_results()

    def preprocess(self) -> None:
        """
        Preprocessing to dataframe before trade.
        """
        return

    def trade(self) -> None:
        """
        Trade signal for each strategy is different.
        The logic should be defined in child class.
        """
        return

    def position_sizing(self) -> float:
        """
        Can override with more sophisticated position sizing approaches.
        """
        return 0.01

    @staticmethod
    def linear(x, a, b):
        return a * x + b

    @staticmethod
    def quadratic(x, a, b):
        return a * x**2 + b * x

    def fit_func(self, x: np.array, y: np.array, degree: int) -> np.ndarray:
        """
        Fit a simple function
        """
        if self.degree == 1:
            self.func = Backtest().linear
        elif degree == 2:
            self.func = Backtest().quadratic
        popt, pcov = curve_fit(self.func, x, y)
        return popt

    def buy(self, row: pd.Series, sizing: bool = True) -> None:
        size, buy_amount = self.shares_buyable(row, sizing)
        self.shares += buy_amount
        shares_value = buy_amount * (row["close"] * 1000)
        self.capital -= shares_value
        self.df.loc[row.name, "action"] = "BUY"
        self.df.loc[row.name, "sizing"] = size
        self.df.loc[row.name, "shares_buyable"] = buy_amount
        self.df.loc[row.name, "fees"] = (self.transaction_fee) * shares_value

    def sell(self, row: pd.Series, sizing: bool = False) -> None:
        size, sell_amount = self.shares_sellable(row, sizing)
        shares_value = sell_amount * (row["close"] * 1000)
        self.capital += shares_value
        self.shares -= sell_amount
        self.df.loc[row.name, "action"] = "SELL"
        self.df.loc[row.name, "sizing"] = size
        self.df.loc[row.name, "shares_sellable"] = sell_amount
        self.df.loc[row.name, "fees"] = (
            self.tax_rate + self.transaction_fee
        ) * shares_value

    def hold(self, row) -> None:
        self.df.loc[row.name, "action"] = "HOLD"
        self.df.loc[row.name, "sizing"] = 0
        self.df.loc[row.name, "fees"] = 0

    def shares_buyable(self, row: pd.Series, sizing: bool = True) -> int:
        size = self.position_sizing() if sizing else 1
        return size, np.floor(size * self.capital / (row["close"] * 1000))

    def shares_sellable(self, row: pd.Series, sizing: bool = False) -> int:
        size = self.position_sizing() if sizing else 1
        return size, np.floor(size * self.shares)

    def calculate_equity(self, row: pd.Series) -> None:
        self.df.loc[row.name, "equity"] = (
            self.shares * row["close"] * 1000 + self.capital
        ) - self.df.loc[row.name, "fees"]

    @staticmethod
    def resample_returns(returns: float, n: int = 1, m: int = 1) -> float:
        return (1 + returns) ** (n / m) - 1

    def preprocess_for_plotting(self) -> None:
        action_onehot = self.df["close"].multiply(pd.get_dummies(self.df["action"]).T).T
        action_onehot[action_onehot == 0] = np.nan
        self.df = pd.concat([self.df, action_onehot], axis=1)

    def plot_results(self) -> None:
        self.preprocess_for_plotting()
        fig = go.Figure()
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["SELL"],
                mode="markers",
                name="SELL",
                marker=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["BUY"],
                mode="markers",
                name="BUY",
                marker=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["close"],
                mode="lines",
                name="close",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["returns"],
                mode="lines",
                name="returns",
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["accum_returns"],
                mode="lines",
                name="accum_returns",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=self.LOG_NAME.replace("_", " "),
            xaxis_title="Date",
            yaxis_title="Closing Price",
            yaxis2_title="Returns",
            height=750,
            width=1250,
            legend=dict(
                x=0, y=1, traceorder="normal", orientation="v", bgcolor="rgba(0,0,0,0)"
            ),
        )

        fig.write_html(self.PLOT)
        if not self.backtest:
            slack.send_file(self.PLOT)

    def consolidate_results(self) -> None:
        self.df["returns"] = self.df["equity"] / self.df["equity"].shift() - 1
        self.df["accum_returns"] = self.df["equity"] / self.initial_capital - 1
        action_values = (
            self.df.groupby(by=["action"]).agg({"returns": "mean"}).to_dict()["returns"]
        )
        total_returns = self.df["accum_returns"][-1]
        annualized_returns = self.__class__.resample_returns(
            total_returns, n=250, m=len(self.df)
        )
        self.df.to_csv(self.RESULTS)
        if not self.backtest:
            last = self.df.iloc[-1]
            slack_message = f"Price: {last['close']}; Recommended action: {last['action']}, Sizing: {last['sizing']} \n (BUY sizing is % of current remaining cash, SELL sizing is % of current shares owned)"
            slack.send_message(str(self.kwargs))
            slack.send_message(
                f"Historical annual returns: {NUM_FORMAT.format(annualized_returns)}"
            )
            slack.send_message(slack_message)
        else:
            messages = [
                str(self.kwargs),
                f"total_returns: {NUM_FORMAT.format(total_returns)}",
                f"annualized_returns: {NUM_FORMAT.format(annualized_returns)}",
                f"action_values: {action_values}",
            ]

            for message in messages:
                write_to_log(self.LOG_NAME, message)
                print(message)
            write_to_log(self.LOG_NAME, "\n")
