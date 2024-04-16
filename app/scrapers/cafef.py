import argparse
import json
import os
import random
import time
import traceback
from datetime import datetime
from typing import List

import pandas as pd

from app.scrapers.base import ScrapeToolbox, write_to_log
from app.scrapers.vndirect import Ticker

START_DATE = "01/01/2000"
END_DATE = datetime.now().date().strftime("%m/%d/%Y")


class StockPrice(ScrapeToolbox):
    LOG_NAME = "cafesp"

    def scrape(self, tickers: List[str], page_size: int = 9999):
        t = 0
        while t < len(tickers):
            try:
                ticker = tickers[t]
                self.start_driver(None)
                self.attack(
                    f"https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol={ticker}&StartDate={START_DATE}&EndDate={END_DATE}&PageIndex=1&PageSize={page_size}",
                    t1=5,
                )
                time.sleep(random.randint(5, 10))
                html = self.attack(
                    self.DRIVER,
                    "//pre",
                    t1=10,
                    t2=5,
                )[
                    0
                ].get_attribute("innerHTML")
                data = self.process_html(html)
                if not data.empty:
                    self.save(ticker, data)
                else:
                    write_to_log(self.LOG_NAME, f"{tickers[t]}: no data obtained")
                t += 1
            except Exception:
                write_to_log(self.LOG_NAME, traceback.format_exc())
            finally:
                self.close_driver()

    def process_html(self, html: str) -> pd.DataFrame:
        html_decoded = json.loads(html)["Data"]["Data"]
        df = pd.DataFrame(html_decoded)
        return df

    def save(self, ticker: str, data: pd.DataFrame) -> None:
        csv_file = f"data/{ticker}_historical_price.csv"
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            data = pd.concat([data, existing_data], axis=0).drop_duplicates()
        data.to_csv(csv_file, index=False)


class OrderStatistic(ScrapeToolbox):
    LOG_NAME = "cafeos"

    def scrape(self, tickers: List[str], page_size: int = 9999):
        t = 0
        while t < len(tickers):
            try:
                ticker = tickers[t]
                self.start_driver(None)
                self.attack(
                    f"https://s.cafef.vn/Ajax/PageNew/DataHistory/ThongKeDL.ashx?Symbol={ticker}&StartDate={START_DATE}&EndDate={END_DATE}&PageIndex=1&PageSize={page_size}",
                    t1=5,
                )
                time.sleep(random.randint(5, 10))
                html = self.attack(
                    self.DRIVER,
                    "//pre",
                    t1=10,
                    t2=5,
                )[
                    0
                ].get_attribute("innerHTML")
                data = self.process_html(html)
                self.save(ticker, data)
                t += 1
            except Exception:
                write_to_log(self.LOG_NAME, traceback.format_exc())
            finally:
                self.close_driver()

    def process_html(self, html: str) -> pd.DataFrame:
        html_decoded = json.loads(html)["Data"]["Data"]
        df = pd.DataFrame(html_decoded)
        return df

    def save(self, ticker: str, data: pd.DataFrame) -> None:
        csv_file = f"data/{ticker}_historical_order.csv"
        if os.path.exists(csv_file):
            existing_data = pd.read_csv(csv_file)
            data = pd.concat([existing_data, data], axis=0).drop_duplicates(keep="last")
        data.to_csv(csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scraper",
        help="""Which scraper to run: StockPrice or OrderStatistic
        """,
        choices=["StockPrice", "OrderStatistic"],
        type=str,
        required=True,
    )

    parser.add_argument(
        "--page_size",
        help="""How much of the most recent data?
        """,
        type=int,
    )
    parser.add_argument("--tickers", nargs="+", default=[])

    args = parser.parse_args()
    if not args.tickers:
        tickers = Ticker().read_tickers()
    else:
        tickers = args.tickers

    scraper = args.scraper
    class_object = globals()[scraper]
    class_object().scrape(tickers, args.page_size)
