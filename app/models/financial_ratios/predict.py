# reference: notebook/XI_fundamentals_analysis.ipynb
import argparse
import pickle
import warnings
from datetime import datetime
from typing import List

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.historical_price import HistoricalPrice
from app.utils import write_to_log

warnings.filterwarnings("ignore")


LOG_NAME = "financial_ratios_model"


def get_financial_infos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "ticker"])
    df.loc[df["Net Profit After Tax"] == 0, "Net Profit After Tax"] = 0.00001
    write_to_log(LOG_NAME, "Financial statements info obtained.")
    return df


def working_capital(row):
    return row["Current Assets"] / row["Short term Liabilities"]


def return_on_equity(row):
    return row["Net Profit After Tax"] / row["TOTAL EQUITY"]


def debt_to_equity(row):
    return row["Liabilities"] / row["TOTAL EQUITY"]


def cash_per_share(row):
    return row["Net cashflow from operating activities"] / (
        row["Net Profit After Tax"] / row["Earnings  per share"]
    )


def get_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["RETURN_ON_EQUITY"] = df.apply(lambda x: return_on_equity(x), axis=1)
    df["DEBT_TO_EQUITY"] = df.apply(lambda x: debt_to_equity(x), axis=1)
    df["WORKING_CAPITAL"] = df.apply(lambda x: working_capital(x), axis=1)
    df["CASH_PER_SHARE"] = df.apply(lambda x: cash_per_share(x), axis=1)
    write_to_log(LOG_NAME, "Financial ratios obtained.")
    return df


def pca_reduce(df: pd.DataFrame) -> pd.DataFrame:
    assert df.iloc[:, :-4].shape[1] == 576
    pca = pickle.load(open("app/models/financial_ratios/pca.pkl", "rb"))
    data = pca.transform(df.iloc[:, :-4].fillna(0))
    data = pd.DataFrame(
        data,
        columns=[f"component_{i+1}" for i in range(data.shape[1])],
    )

    reduced_df = pd.DataFrame(data=data.values, columns=data.columns, index=df.index)
    reduced_df = pd.concat([df.iloc[:, -4:], reduced_df], axis=1)
    write_to_log(LOG_NAME, "Principal components obtained.")
    return reduced_df


def get_close(df: pd.DataFrame) -> pd.DataFrame:
    adj = pd.DataFrame(columns=["date", "close", "ticker"])
    hp = HistoricalPrice()
    tickers = df.index.remove_unused_levels().levels[1]

    for ticker in tickers:
        try:
            close = hp.get_asset_price(ticker, "monthly")["close"].reset_index()
            close["ticker"] = ticker
            adj = pd.concat([adj, close])
        except Exception:
            continue

    adj["date"] = pd.to_datetime(adj["date"])
    adj = adj.set_index(["date", "ticker"])
    df = df.merge(adj, left_index=True, right_index=True, how="outer")
    df = df.dropna().drop_duplicates()
    write_to_log(LOG_NAME, "Closing prices obtained.")
    return df


def get_lags(df: pd.DataFrame) -> pd.DataFrame:
    lag_df = pd.DataFrame()
    train_columns = df.columns
    tickers = df.index.remove_unused_levels().levels[1]

    for ticker in tickers:
        try:
            t = df.loc[(slice(None), ticker), :].sort_index(ascending=True)
            for i in range(6):
                for col in train_columns:
                    t[col + f"_lag{i+1}"] = t[col].shift(i + 1)
            t["target"] = t["close"].shift(-1)
            lag_df = pd.concat([lag_df, t], axis=0)
        except Exception:
            continue

    shape_before = lag_df.shape
    lag_df = lag_df.dropna(subset=lag_df.columns.difference(["target"]), how="any")
    write_to_log(
        LOG_NAME,
        f"Lag features obtained, {lag_df.shape[1]-shape_before[1]} rows with missing data were dropped.",
    )
    return lag_df


def get_last_reporting_date() -> pd.DataFrame:
    date_format = "%Y-%m-%d"
    today = datetime.now().date()
    reporting_dates = [
        "03-31",
        "06-30",
        "09-30",
        "12-31",
    ]

    reporting_dates = [
        datetime.strptime(f"{today.year}-{d}", date_format).date()
        for d in reporting_dates
    ]

    differences = [(date - today).days for date in reporting_dates]
    for index, diff in enumerate(differences):
        if diff > 0:
            if index != 0:
                return reporting_dates[index - 1].strftime(date_format)
            elif index == 0:
                return (
                    datetime.strptime(f"{today.year-1}-12-31", date_format)
                    .date()
                    .strftime(date_format)
                )


def predict(test_X):
    rfr = pickle.load(
        open(
            f"app/models/financial_ratios/rf_regressor_{args.industry.replace(' ','_').lower()}.pkl",
            "rb",
        )
    )
    pred = rfr.predict(test_X)
    return pred


def compute_acc(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["target"])
    pred = df["pred"].values
    target = df["target"].values

    mse = mean_squared_error(pred, target)
    mae = mean_absolute_error(pred, target)
    message = f"MAE: {mae}, MSE: {mse}"
    write_to_log(LOG_NAME, message)
    print(message)


def save_output(df: pd.DataFrame) -> None:
    out = df.loc[:, ["close", "pred", "target"]]
    out.to_csv(
        f"app/models/financial_ratios/financial_ratios_model_pred_{args.industry.replace(' ','_').lower()}.csv"
    )


def get_industry_constituents(selected: str) -> List[str]:
    industry = pd.read_csv("data/tickers.csv", index_col=0).loc[
        :, ["industry", "floor", "sub_industry"]
    ]
    selected_industry = industry[industry["industry"] == selected].index.tolist()
    return selected_industry


def main():
    constituents = get_industry_constituents(args.industry)

    last = get_last_reporting_date()
    write_to_log(LOG_NAME, "#### INITIATE ####")
    write_to_log(LOG_NAME, f"Last reporting date: {last}")
    path = "data/financial_statement_rearranged.csv"
    df = get_financial_infos(path)
    df = df.loc[df.index.get_level_values("ticker").isin(constituents)]
    df = get_ratios(df)

    df = pca_reduce(df)
    df = get_close(df)
    df = get_lags(df)

    X = df.drop("target", axis=1)
    pred = predict(X)
    df["pred"] = pred
    compute_acc(df)
    save_output(df)


# python3 -m app.models.financial_ratios.predict --industry "Health Care"
# python3 -m app.models.financial_ratios.predict --industry "Food & Beverage"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", type=str, required=True)
    args = parser.parse_args()
    main()
