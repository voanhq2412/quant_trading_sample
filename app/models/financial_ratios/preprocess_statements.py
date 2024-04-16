import glob

import pandas as pd


def read_statements() -> pd.DataFrame:
    files = glob.glob("data/vnd_financial_statement*.csv")
    for index, file in enumerate(files):
        df = pd.read_csv(file, index_col=None).filter(
            items=["ticker", "name", "value", "date"]
        )
        if index == 0:
            data = df.copy()
        else:
            data = pd.concat([data, df.copy()], axis=0)
    return data


def rearrange_statements(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(by=["ticker", "name", "date"], ascending=False)
    data = data.drop_duplicates().dropna(subset=["name"]).reset_index(drop=True)

    # SORT
    tickers = list(sorted(set(data["ticker"])))
    names = list(sorted(set(data["name"])))
    dates = list(sorted(set(data["date"]), reverse=True))

    # CREATE dataframe, and fill data
    index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["date", "ticker"])
    df = pd.DataFrame(columns=names, index=index)

    for ticker in tickers:
        for date in dates:
            ticker_data = data.loc[
                (data["ticker"] == ticker) & (data["date"] == date), ["name", "value"]
            ]
            ticker_data = ticker_data.set_index("name").T
            ticker_data["date"] = date
            ticker_data["ticker"] = ticker
            ticker_data = ticker_data.set_index(["date", "ticker"])
            ticker_data = ticker_data.loc[:, ~ticker_data.columns.duplicated()].copy()

            if ticker_data.empty:
                continue
            else:
                df = pd.concat([df, ticker_data], axis=0)
                print(df.shape)
    df.to_csv("data/financial_statement_rearranged.csv")


#  python3 -m app.models.financial_ratios.preprocess_statements
if __name__ == "__main__":
    df = read_statements()
    rearrange_statements(df)
