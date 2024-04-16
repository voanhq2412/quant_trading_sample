import glob
import re
from sys import platform
from typing import Dict, List

import pandas as pd
from scipy import stats

pd.set_option("display.max_rows", None)


class HistoricalPrice:
    # Minimum 4 years data total
    FREQ = {
        "monthly": {"min_sample": 12 * 4, "resample": "M"},
        "fortnightly": {"min_sample": 26 * 4, "resample": "2W-MON"},
        "weekly": {"min_sample": 52 * 4, "resample": "W-MON"},
        "daily": {"min_sample": 365 * 4, "resample": "D"},
        "quarterly": {"min_sample": 4 * 4, "resample": "Q"},
    }

    def path(self) -> str:
        if platform == "linux":
            return "data/*historical_price.csv"
        elif platform == "win32":
            return "data\*historical_price.csv"

    def regex(self) -> str:
        if platform == "linux":
            return r"(?<=\/)(.[A-Z0-9]{2})(?=_historical)"
        elif platform == "win32":
            return r"(?<=\\)(.[A-Z0-9]{2})(?=_historical)"

    def get_historical_prices(self) -> Dict[str, str]:
        paths = [file for file in glob.glob(self.path())]
        return {
            re.search(self.regex(), path).group(0): path
            for path in paths
            if re.search(self.regex(), path)
        }

    def get_order_info(self, ticker: str) -> pd.DataFrame:
        order = pd.read_csv(f"data/{ticker}_historical_order.csv")
        order["Date"] = pd.to_datetime(order["Date"], format="%d/%m/%Y")
        order = order.set_index("Date").drop(columns="ThayDoi")
        order["ChenhLechKL"] = order["KLDatMua"] - order["KLDatBan"]
        return order

    def get_asset_price(self, ticker: str, freq: str) -> pd.DataFrame:
        df = self.read(ticker)
        df = df.resample(self.FREQ[freq]["resample"], convention="end").agg(
            {
                "low": "last",
                "high": "last",
                "open": "last",
                "close": "last",
                "adj_close": "last",
                "order_matching_volume": "sum",
                "order_matching_value": "sum",
                "order_negotiated_volume": "sum",
                "order_negotiated_value": "sum",
            }
        )
        return df

    def get_returns(self, freq: str, tickers: List[str] = []) -> pd.DataFrame:
        self.min_sample = self.FREQ[freq]["min_sample"]
        returns = None

        if not tickers:
            tickers = self.get_historical_prices().keys()

        for ticker in tickers:
            try:
                asset_returns = self.returns(ticker, freq)
                temp_df = pd.DataFrame(
                    data=asset_returns["r"].values,
                    columns=[ticker],
                    index=asset_returns.index,
                )

                if temp_df.notna().sum().values[0] >= self.min_sample:
                    # ensure adequate sample size
                    if returns is not None:
                        returns = pd.concat([temp_df, returns], axis=1)
                    else:
                        returns = temp_df
            except Exception:
                continue
        return returns.resample(self.FREQ[freq]["resample"]).last().dropna(how="all")

    def get_corr(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        corr = {}
        for r1 in df1.columns:
            for r2 in df2.columns:
                if r1 != r2:
                    temp = (
                        pd.concat([df1[r1], df2[r2]], axis=1)
                        .dropna()
                        .sort_index(axis=1)
                    )
                    if len(temp) >= self.min_sample:
                        tau, p_value = stats.kendalltau(temp[r1], temp[r2])
                        if (r1 + "_" + r2) not in corr:
                            corr[r1 + "_" + r2] = [tau, p_value]
        return corr

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [
            "date",
            "adj_close",
            "close",
            "change",
            "order_matching_volume",
            "order_matching_value",
            "order_negotiated_volume",
            "order_negotiated_value",
            "open",
            "high",
            "low",
        ]
        df = df.drop(columns=["change"])
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df["volume"] = df["order_matching_volume"] + df["order_negotiated_volume"]
        df = df.set_index("date")
        df = df.sort_index(ascending=True)
        return df

    def read(self, ticker: str) -> pd.DataFrame:
        df = pd.read_csv(f"data/{ticker}_historical_price.csv")
        return self.preprocess(df)

    def returns(self, ticker: str, freq: str) -> pd.DataFrame:
        df = self.get_asset_price(ticker, freq)
        df["day"] = df.index.dayofweek
        df = df.loc[df["day"] < 5]
        df["r"] = df["adj_close"] / df["adj_close"].shift() - 1
        return df.dropna()
