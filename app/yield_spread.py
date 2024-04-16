import json

import pandas as pd
import requests

api_key = "de4b58a36d3e5c453374ad31cbbab336"
file_type = "json"
category_id = "T10Y3MM"

url = f"https://api.stlouisfed.org/fred/series/observations?series_id={category_id}&frequency=d&aggregation_method=eop&api_key={api_key}&file_type={file_type}"


def get_us_yield():
    response = requests.get(url)
    data = json.loads(response.content)

    df = pd.DataFrame.from_dict(data["observations"])
    df = df.filter(items=["date", "value"])
    df = df.set_index("date").loc["2000-07-01":]
    df["value"] = df["value"].astype("float32")
    return df
