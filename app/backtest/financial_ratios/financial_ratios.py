import argparse
import warnings

import pandas as pd

from app import slack
from app.backtest.backtest import Backtest
from app.historical_price import HistoricalPrice
from app.scrapers.vndirect import Ticker
from app.utils import get_current_dir

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
CURRENT_DIR = get_current_dir()


class FinancialRatiosStrategy(Backtest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.LOG_NAME = f"financial_ratios_backtest_{self.ticker}"
        self.RESULTS = f"{CURRENT_DIR}/{self.ticker}.csv"
        self.PLOT = f"{CURRENT_DIR}/{self.ticker}.html"

    def position_sizing(self) -> float:
        return self.max_portion

    def preprocess(self) -> None:
        if self.degree != 0:
            xy = self.df.loc[:, ["pred", "target"]].dropna()
            x = xy["pred"].values
            y = xy["target"].values
            popt = self.fit_func(x, y, self.degree)

            self.df["pred"] = self.func(self.df["pred"].values, *popt)
        return

    def trade(self, row: pd.Series) -> None:
        if row["pred"] > (1 + self.multiplier) * row["close"]:
            self.buy(row)
        elif row["pred"] < (1 - self.multiplier) * row["close"]:
            self.sell(row)
        else:
            self.hold(row)

        self.calculate_equity(row)


def get_ticker_pred(ticker: str) -> pd.DataFrame:
    path = f"app/models/financial_ratios/financial_ratios_model_pred_{args.industry.replace(' ','_').lower()}.csv"
    df = pd.read_csv(path)
    df = df.set_index("date")
    df = df.loc[df["ticker"] == ticker, ["pred", "target"]]
    df.index = pd.to_datetime(df.index).to_period("Q") + 1
    return df


def get_daily_close(ticker: str) -> pd.DataFrame:
    # FOR FINANCIAL RATIOS STRATEGY, ignore first year data only
    hp = HistoricalPrice()
    price = hp.get_asset_price(ticker, "daily").loc[:, ["close"]].dropna()
    price["quarter"] = price.index.to_period("Q")
    return price.iloc[250:]


def main():
    df = get_ticker_pred(args.ticker)
    price = get_daily_close(args.ticker)
    price = price.merge(df, left_on="quarter", right_index=True, how="outer")
    if not args.backtest:
        try:
            ticker_class = Ticker()
            date, close = ticker_class.get_live_price(args.ticker)
            price.loc[date, "close"] = close
        except IndexError:
            message = f"No live price obtained: {args.ticker}"
            slack.send_message(message)
            return
    else:
        price = price.dropna(subset=["target"])
    FinancialRatiosStrategy(**vars(args)).execute(price)


# TUNED
# python3 -m app.backtest.financial_ratios.financial_ratios --ticker OPC --industry "Health Care" --multiplier 0 --degree 2 --initial_capital 3000000 --max_portion 1 --backtest
# 0.2537

# python3 -m app.backtest.financial_ratios.financial_ratios --ticker VDP --industry "Health Care" --multiplier 0 --degree 2 --initial_capital 3000000 --max_portion 1 --backtest
# 0.4576

# python3 -m app.backtest.financial_ratios.financial_ratios --ticker NSC --industry "Food & Beverage" --multiplier 0 --degree 2 --initial_capital 3000000 --max_portion 1 --backtest
# 0.3177

# python3 -m app.backtest.financial_ratios.financial_ratios --ticker BAF --industry "Food & Beverage" --multiplier 0.11 --degree 2 --initial_capital 3000000 --max_portion 1 --backtest
# 0.5361

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--ticker", required=True, type=str)
    parser.add_argument("--initial_capital", type=float, required=True)
    parser.add_argument("--max_portion", type=float, required=True)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--industry", required=True, type=str)
    parser.add_argument(
        "--degree",
        type=int,
        required=True,
        help="reduce prediction noise by fitting polynomial function",
    )

    args = parser.parse_args()
    main()
