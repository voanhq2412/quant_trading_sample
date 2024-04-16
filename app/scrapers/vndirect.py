import argparse
import json
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz
import requests

from app import slack
from app.utils import write_to_log

warnings.filterwarnings("ignore")


class Ticker:
    DATA = pd.DataFrame(None, index=None, columns=["industry", "sub_industry"])
    TICKERS = "data/tickers.csv"
    HEADERS = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Origin": "https://dstock.vndirect.com.vn",
        "Referer": "https://dstock.vndirect.com.vn/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }

    def read_tickers(self):
        try:
            csv = pd.read_csv(self.TICKERS, index_col=0)
            return csv.index
        except Exception:
            print("Can't read tickers file.")

    def scrape(self):
        self.get_industries()
        self.get_sub_industries()
        df = self.get_company_details()
        self.DATA = pd.merge(self.DATA, df, left_index=True, right_index=True)
        self.DATA.to_csv(self.TICKERS)
        message = "Successfuly obtained tickers with industry and company info"
        slack.send_message(message)

    def get_industries(self):
        response = requests.get(
            "https://api-finfo.vndirect.com.vn/v4/industry_classification?q=industryLevel:2",
            headers=self.HEADERS,
        )

        js = json.loads(response.content)
        for data in js["data"]:
            tickers = data["codeList"].split(",")
            for ticker in tickers:
                self.DATA.loc[ticker, "industry"] = data["englishName"]

    def get_sub_industries(self):
        response = requests.get(
            "https://api-finfo.vndirect.com.vn/v4/industry_classification?q=industryLevel:3",
            headers=self.HEADERS,
        )

        js = json.loads(response.content)
        for data in js["data"]:
            if "codeList" in data:
                tickers = data["codeList"].split(",")
                for ticker in tickers:
                    self.DATA.loc[ticker, "sub_industry"] = data["englishName"]

    def get_company_details(self) -> pd.DataFrame:
        response = requests.get(
            "https://finfo-api.vndirect.com.vn/v4/stocks?q=type:stock,ifc~floor:HOSE,HNX&size=9999",
            headers=self.HEADERS,
        )

        js = json.loads(response.content)
        df = pd.DataFrame.from_dict(js["data"])
        df = df.loc[df["code"].str.len() == 3].set_index("code")
        return df

    def get_live_price(
        self, ticker: str, date: datetime.date = None
    ) -> Tuple[str, np.float]:
        """
        2:00 UTC time = 9:00 AM time
        7:00 UTC time == 14:00 VN time
        Trading session ends at 14:30.

        Obtain live price starting from 13:00 VN time, so 6:00 UTC
        """
        date = datetime.now(timezone.utc) if date is None else date
        utc_time = datetime(
            date.year,
            date.month,
            date.day,
            date.hour,
            0,
            0,
            tzinfo=pytz.timezone("UTC"),
        )

        timestamp = int(utc_time.timestamp())
        response = requests.get(
            "https://dchart-api.vndirect.com.vn/dchart/history",
            params={
                "resolution": "1",
                "symbol": ticker,
                "from": timestamp,
            },
            headers=self.HEADERS,
        )
        df = pd.DataFrame.from_dict(json.loads(response.content))
        try:
            last = df.iloc[-1]
            price, ts = last[["c", "t"]]
            date = datetime.fromtimestamp(ts, tz=pytz.timezone("Asia/Ho_Chi_Minh"))
            message = f"Successfuly obtained live price for {ticker} at {date.strftime('%H:%M %m/%d/%Y')}"
            slack.send_message(message)
            return date.strftime("%m/%d/%Y"), price
        except (IndexError, ValueError) as e:
            print(utc_time, e)
            date = date - timedelta(minutes=10)
            return self.get_live_price(ticker, date)


class FinancialStatement(Ticker):
    LOG_NAME = "vnd_financial_statement"

    def scrape(self, tickers: List[int] = None, years: List[int] = None) -> None:
        if not tickers:
            tickers = Ticker().read_tickers()
        if not args.years:
            years = [y for y in range(2000, datetime.now().year + 1)]

        for ticker in tickers:
            if len(ticker) == 3:
                file_path = f"data/{self.LOG_NAME}_{ticker}.csv"
                self.init_dataframe(file_path)
                print(self.data.shape)
                self.get_item_codes(ticker)
                print(self.item_codes)
                if not self.item_codes:
                    continue
                else:
                    self.get_item_values(ticker, years)
                self.write_data(file_path)
            time.sleep(5)

        message = "Successfuly obtained financial statements"
        slack.send_message(message)

    def init_dataframe(self, file_path: str) -> None:
        file = Path(file_path)
        if file.exists():
            self.data = pd.read_csv(file_path, index_col=False)
        else:
            self.data = pd.DataFrame(
                None, index=None, columns=["ticker", "name", "code", "value", "date"]
            )

    def write_data(self, file_path: str) -> None:
        self.data.to_csv(file_path, index=False)

    def get_quarters_string(self, years: List[int]) -> str:
        quarters_string = ""
        today = datetime.now()
        for year in years:
            for date in ["12-31", "09-30", "06-30", "03-31"]:
                date_string = str(year) + "-" + date
                dt = datetime.strptime(date_string, "%Y-%m-%d")
                if dt < today:
                    quarters_string += date_string + ","

        return quarters_string[:-1]

    def get_item_codes(self, ticker: str) -> None:
        statements = {
            "balance_sheet_items": f"https://finfo-api.vndirect.com.vn/v4/financial_models?sort=displayOrder:asc&q=codeList:{ticker}~modelType:1,89,101,411~note:TT199/2014/TT-BTC,TT334/2016/TT-BTC,TT49/2014/TT-NHNN,TT202/2014/TT-BTC~displayLevel:0,1,2,3&size=1000",
            "income_statement_items": f"https://finfo-api.vndirect.com.vn/v4/financial_models?sort=displayOrder:asc&q=codeList:{ticker}~modelType:2,90,102,412~note:TT199/2014/TT-BTC,TT334/2016/TT-BTC,TT49/2014/TT-NHNN,TT202/2014/TT-BTC~displayLevel:0,1,2,3&size=1000",
            "cash_flow_items": f"https://finfo-api.vndirect.com.vn/v4/financial_models?sort=displayOrder:asc&q=codeList:{ticker}~modelType:3,91,103,413~note:TT199/2014/TT-BTC,TT334/2016/TT-BTC,TT49/2014/TT-NHNN,TT202/2014/TT-BTC~displayLevel:0,1,2,3&size=1000",
        }
        self.item_codes = {}
        for statement, url in statements.items():
            response = requests.get(url, headers=self.HEADERS)
            try:
                js = json.loads(response.content)
                for data in js["data"]:
                    self.item_codes[data["itemCode"]] = data["itemEnName"]
            except json.decoder.JSONDecodeError:
                write_to_log(
                    self.LOG_NAME, f"Failed to obtain {statement} for {ticker}"
                )
                continue

    def get_item_values(self, ticker: str, years: List[int]) -> None:
        quarters_string = self.get_quarters_string(years)
        balance_sheet_values = f"https://finfo-api.vndirect.com.vn/v4/financial_statements?q=code:{ticker}~reportType:QUARTER~modelType:1,89,101,411~fiscalDate:{quarters_string}&sort=fiscalDate&size=9999"
        income_statement_values = f"https://finfo-api.vndirect.com.vn/v4/financial_statements?q=code:{ticker}~reportType:QUARTER~modelType:2,90,102,412~fiscalDate:{quarters_string}&sort=fiscalDate&size=9999"
        cash_flow_values = f"https://finfo-api.vndirect.com.vn/v4/financial_statements?q=code:{ticker}~reportType:QUARTER~modelType:3,91,103,413~fiscalDate:{quarters_string}&sort=fiscalDate&size=9999"

        values = [balance_sheet_values, income_statement_values, cash_flow_values]

        for value in values:
            response = requests.get(
                value,
                headers=self.HEADERS,
            )
            try:
                js = json.loads(response.content)
                for data in js["data"]:
                    self.get_item(ticker, data)
            except json.decoder.JSONDecodeError:
                write_to_log(
                    self.LOG_NAME, f"Failed to obtain {value} for {ticker} for {years}"
                )
                continue

    def get_item(self, ticker, data) -> None:
        try:
            row = {}
            row["ticker"] = ticker
            row["name"] = self.item_codes[data["itemCode"]]
            row["code"] = data["itemCode"]
            row["value"] = data["numericValue"]
            row["date"] = data["fiscalDate"]
            self.data = self.data.append(row, ignore_index=True)
        except Exception:
            pass


# etc
# python3 -m app.scrapers.vndirect --scraper FinancialStatement --start 2023

# or
# ipython
# from app.scrapers.vndirect import Ticker
# Ticker().get_live_price('VIC')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scraper",
        help="""Which scraper to run: VnDirect (tickers, company details, industry and live price),
        and FinancialStatement.
        """,
        choices=["Ticker", "FinancialStatement"],
        type=str,
        required=True,
    )
    parser.add_argument(
        "--years",
        help="Scrape financial statements for specific years.",
        nargs="+",
        default=[],
        required=False,
    )
    parser.add_argument(
        "--tickers",
        help="Scrape financial statements for specific scrapers.",
        nargs="+",
        default=[],
        required=False,
    )

    args = parser.parse_args()
    scraper = args.scraper
    class_object = globals()[scraper]

    if scraper == "FinancialStatement":
        class_object().scrape(args.tickers, args.years)
    else:
        class_object().scrape()


# import requests

# headers = {
#     "Accept": "*/*",
#     "Accept-Language": "en-US,en;q=0.9",
#     "Connection": "keep-alive",
#     "Origin": "https://dstock.vndirect.com.vn",
#     "Referer": "https://dstock.vndirect.com.vn/",
#     "Sec-Fetch-Dest": "empty",
#     "Sec-Fetch-Mode": "cors",
#     "Sec-Fetch-Site": "same-site",
#     "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
#     "sec-ch-ua-mobile": "?0",
#     "sec-ch-ua-platform": '"Linux"',
# }

# response = requests.get(
#     "https://finfo-api.vndirect.com.vn/v4/financial_models?sort=displayOrder:asc&q=codeList:ACB~modelType:1,89,101,411~note:TT199/2014/TT-BTC,TT334/2016/TT-BTC,TT49/2014/TT-NHNN,TT202/2014/TT-BTC~displayLevel:0,1,2,3&size=999",
#     headers=headers,
# )
# print(response.content)
