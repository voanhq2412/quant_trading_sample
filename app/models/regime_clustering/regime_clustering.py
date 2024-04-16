import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import plotly.graph_objects as go
import plotly.subplots as sp
from hmmlearn import hmm

from app.historical_price import HistoricalPrice

logging.getLogger("hmmlearn").setLevel("CRITICAL")


class GMMHMM:
    def set_column_features(self, df: pd.DataFrame):
        self.cols = [i for i in df.columns if "direction" in i]

    def feature_engineer(self, df: pd.DataFrame, lag: int) -> pd.DataFrame:
        """
        USE GMMHMM to cluster states based on directional change.
        """
        df = df.dropna(subset=["close"])
        df = self.get_direction(df, lag)
        self.set_column_features(df)
        return df

    def point_predict(self, val: float) -> pd.DataFrame:
        return self.model.predict(np.array(val).reshape(1, -1))

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df["state"] = self.model.predict(df.loc[:, self.cols])
        return df

    def post_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create two columns, one for each state. y
        """
        states = pd.get_dummies(df["state"])
        states.columns = ["state_0", "state_1"]
        states_onehot = df["close"].multiply(states.T).T
        states_onehot[states_onehot == 0] = np.nan
        df = pd.concat([df, states_onehot], axis=1)
        df.index = pd.DatetimeIndex(df.index).to_period("D")
        df.index = df.index.to_timestamp()
        return df

    def get_direction(self, df: pd.DataFrame, lag: int) -> pd.DataFrame:
        """
        Get weighted moving average of the last 'lag' points.
        Obtain lagged returns and directional change.
        """
        df[f"ma_{lag}"] = df.ta.wma(lag)
        df[f"returns_{lag}"] = df["close"] / df[f"ma_{lag}"].shift(lag) - 1
        df[f"directional_change_{lag}"] = df[f"returns_{lag}"] / np.abs(
            df[f"returns_{lag}"]
        )
        return df.fillna(method="bfill").dropna()

    def get_best_model(self, df: pd.DataFrame) -> None:
        """
        Fit GMMHMM 10000 times and pick the best model.
        (minimize randomness)
        """
        best_mle = -100000
        best_model = None
        for i in range(100):
            try:
                model = hmm.GMMHMM(n_components=2, n_iter=10000)
                model.fit(df.loc[:, self.cols])
                mle = model.score(df.loc[:, self.cols])
                if mle > best_mle:
                    best_mle = mle
                    best_model = model
            except Exception:
                continue
        self.model = best_model

    def save_plot(self, df: pd.DataFrame, plot_path: str) -> None:
        df = self.post_processing(df)
        fig = go.Figure()
        fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["state_0"], mode="markers", marker=dict(color="blue")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["state_1"], mode="markers", marker=dict(color="red")
            )
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Closing Prices",
            height=750,
            width=1250,
        )
        fig.write_html(plot_path)

    def save_model(self, path: str) -> None:
        pickle.dump(self.model, open(path, "wb"))

    def load_model(self, path: str) -> None:
        self.model = pickle.load(open(path, "rb"))


def cluster(ticker: str, lag: int) -> pd.DataFrame:
    hp = HistoricalPrice()
    df = hp.get_asset_price(ticker, "daily")

    model_path = f"app/models/regime_clustering/{ticker}_{lag}.pkl"
    plot_path = f"app/models/regime_clustering/{ticker}_{lag}.html"

    gmmhmm = GMMHMM()
    df = gmmhmm.feature_engineer(df, lag)

    if os.path.exists(model_path):
        gmmhmm.load_model(model_path)
    else:
        gmmhmm.get_best_model(df)
        gmmhmm.save_model(model_path)

    clustered_df = gmmhmm.batch_predict(df)
    gmmhmm.save_plot(clustered_df, plot_path)
    return clustered_df


def main():
    clustered_df = cluster(args.ticker, args.lag)
    return clustered_df


# python3 -m app.models.regime_clustering.regime_clustering --ticker DRH --lag 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker",
        help="Which stock?",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lag",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    # main()
