import argparse
import calendar
import warnings
from typing import List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from app import slack
from app.backtest.backtest import Backtest
from app.historical_price import HistoricalPrice
from app.models.regime_clustering.regime_clustering import cluster
from app.scrapers.vndirect import Ticker
from app.utils import get_current_dir

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)

CURRENT_DIR = get_current_dir()


class CorrelatedPairStrategy(Backtest):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.LOG_NAME = f"correlated_pair_backtest_{'_'.join(self.pair)}"
        self.RESULTS = f"{CURRENT_DIR}/{'_'.join(self.pair)}.csv"
        self.PLOT = f"{CURRENT_DIR}/{'_'.join(self.pair)}.html"

        self.deviation = []
        self.week = None
        self.days_past = None

    def position_sizing(self) -> float:
        return min((np.abs(self.deviation[-1]) / self.max_dev), self.max_portion)

    def preprocess(self) -> None:
        hp = HistoricalPrice()
        weekly_returns = hp.get_returns(freq="weekly", tickers=self.pair)
        x, y = (
            weekly_returns.loc[:, self.pair]
            .sort_values(by=[self.pair[0]], ascending=True)
            .dropna()
            .values
        ).T
        self.popt = self.fit_func(x, y, self.degree)

    def get_key_dates(self, row: pd.Series) -> Tuple:
        today = row.name
        year = today.year
        month = str(today.month).zfill(2)
        days_in_month = calendar.monthrange(today.year, today.month)[1]

        first_trading_day = self.df.loc[f"{year}-{month}-01":, :].index[0]
        today = today.strftime("%Y-%m-%d")
        temp = self.df.loc[first_trading_day:today, :]

        days_past = len(temp) - 1
        return first_trading_day, today, days_in_month, days_past


class test(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.buy(row, sizing=True)
                else:
                    self.buy(row, sizing=True)
        self.calculate_equity(row)


class CTG_HDB(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 1 or row["state_5"] == 1 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class MBB_VND(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 1 or row["state_5"] == 1 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


# class STB_LPB(CorrelatedPairStrategy):
#     def trade(self, row: pd.Series) -> None:
#         today = row.name

#         if self.week is None or self.week != self.df.loc[today, "week"]:
#             """
#             START NEW WEEK
#             - Reset days_past
#             - Use last week closing as this week opening
#             """
#             first_date = today.strftime("%Y-%m-%d")
#             self.week = self.df.loc[today, "week"]
#             self.days_past = 1
#             self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
#             self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
#         else:
#             self.days_past += 1

#         return_x = super().resample_returns(
#             returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
#             - 1,
#             m=self.days_past,
#         )

#         return_y = super().resample_returns(
#             returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
#             - 1,
#             m=self.days_past,
#         )

#         weekly_return_x = (1 + return_x) ** (5) - 1
#         pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

#         pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

#         dev = pred_return_y - return_y
#         self.deviation.append(dev)

#         # Y is over-valued by a factor
#         if return_y >= self.multiplier * pred_return_y:
#             if return_y > 0 and (
#                 row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
#             ):
#                 if return_x > 0:
#                     self.hold(row)
#                 else:
#                     self.sell(row)
#             else:
#                 if return_x > 0:
#                     self.hold(row)
#                 else:
#                     self.sell(row)

#         # Y is undervalued by a factor
#         elif return_y < self.multiplier * pred_return_y:
#             if return_y > 0 and (
#                 row["state_3"] == 1 or row["state_5"] == 1 or row["state_200"] == 1
#             ):
#                 if return_x > 0:
#                     self.buy(row, sizing=False)
#                 else:
#                     self.hold(row)
#             else:
#                 if return_x > 0:
#                     self.hold(row)
#                 else:
#                     self.hold(row)
#         self.calculate_equity(row)


class VCI_FTS(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 1 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 1 or row["state_20"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class VCI_CTS(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 1 or row["state_5"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_20"] == 1 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class MBS_BSI(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name
        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.buy(row, sizing=True)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class CTS_FTS(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class VGS_TLH(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
        self.calculate_equity(row)


class VCG_DIG(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
            ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0 and (
                row["state_3"] == 1 or row["state_20"] == 1 or row["state_200"] == 1
            ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.buy(row, sizing=False)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
        self.calculate_equity(row)


class PLX_PVS(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        today = row.name

        if self.week is None or self.week != self.df.loc[today, "week"]:
            """
            START NEW WEEK
            - Reset days_past
            - Use last week closing as this week opening
            """
            first_date = today.strftime("%Y-%m-%d")
            self.week = self.df.loc[today, "week"]
            self.days_past = 1
            self.week_open_pair_0 = self.df.loc[first_date, f"{self.pair[0]}_open"]
            self.week_open_pair_1 = self.df.loc[first_date, f"{self.pair[1]}_open"]
        else:
            self.days_past += 1

        return_x = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[0]}_close"] / self.week_open_pair_0
            - 1,
            m=self.days_past,
        )

        return_y = super().resample_returns(
            returns=self.df.loc[today, f"{self.pair[1]}_close"] / self.week_open_pair_1
            - 1,
            m=self.days_past,
        )

        weekly_return_x = (1 + return_x) ** (5) - 1
        pred_weekly_return_y = self.func(weekly_return_x, *self.popt)

        pred_return_y = super().resample_returns(returns=pred_weekly_return_y, m=5)

        dev = pred_return_y - return_y
        self.deviation.append(dev)

        # Y is over-valued by a factor
        if return_y >= self.multiplier * pred_return_y:
            if return_y > 0:  # and (
                # row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 1
                # ):
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)
            else:
                if return_x > 0:
                    self.hold(row)
                else:
                    self.sell(row)

        # Y is undervalued by a factor
        elif return_y < self.multiplier * pred_return_y:
            if return_y > 0:  # and (
                # row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 0
                # ):
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
            else:
                if return_x > 0:
                    self.buy(row, sizing=False)
                else:
                    self.hold(row)
        self.calculate_equity(row)


class PLP_DRH(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        first_trading_day, today, days_in_month, days_past = self.get_key_dates(row)
        if days_past > 0:
            return_x = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[0]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[0]}_close"]
                - 1,
                m=days_past,
            )

            return_y = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[1]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[1]}_close"]
                - 1,
                m=days_past,
            )

            monthly_return_x = super().resample_returns(
                returns=return_x, n=days_in_month - 1
            )
            pred_monthly_return_y = self.func(monthly_return_x, *self.popt)

            pred_return_y = super().resample_returns(
                returns=pred_monthly_return_y, m=days_in_month - 1
            )
            dev = pred_return_y - return_y
            self.deviation.append(dev)

            # Y is over-valued by a factor
            if return_y >= self.multiplier * pred_return_y:
                if return_y > 0 and (
                    row["state_3"] == 0 or row["state_5"] == 1 or row["state_200"] == 0
                ):
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)

            # Y is undervalued by a factor
            elif return_y < self.multiplier * pred_return_y:
                if return_y > 0 and (
                    row["state_3"] == 0 or row["state_20"] == 1 or row["state_200"] == 0
                ):
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.buy(row, sizing=False)

        else:
            self.hold(row)
        self.calculate_equity(row)


class PDR_MBS(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        first_trading_day, today, days_in_month, days_past = self.get_key_dates(row)
        if days_past > 0:
            return_x = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[0]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[0]}_close"]
                - 1,
                m=days_past,
            )

            return_y = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[1]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[1]}_close"]
                - 1,
                m=days_past,
            )
            monthly_return_x = super().resample_returns(
                returns=return_x, n=days_in_month - 1
            )
            pred_monthly_return_y = self.func(monthly_return_x, *self.popt)

            pred_return_y = super().resample_returns(
                returns=pred_monthly_return_y, m=days_in_month - 1
            )
            dev = pred_return_y - return_y
            self.deviation.append(dev)

            # Y is over-valued by a factor
            if return_y >= self.multiplier * pred_return_y:
                if return_y > 0:  # and (row["state_3"] == 0 or row["state_5"] == 1):
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)

            # Y is undervalued by a factor
            elif return_y < self.multiplier * pred_return_y:
                if return_y > 0:  # and (row["state_3"] == 1 or row["state_20"] == 0):
                    if return_x > 0:
                        # self.buy(row)
                        self.buy(row, sizing=False)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        # self.buy(row)
                        self.buy(row, sizing=False)
        else:
            self.hold(row)
        self.calculate_equity(row)


class HAP_EVG(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        first_trading_day, today, days_in_month, days_past = self.get_key_dates(row)
        if days_past > 0:
            return_x = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[0]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[0]}_close"]
                - 1,
                m=days_past,
            )

            return_y = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[1]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[1]}_close"]
                - 1,
                m=days_past,
            )

            monthly_return_x = super().resample_returns(
                returns=return_x, n=days_in_month - 1
            )
            pred_monthly_return_y = self.func(monthly_return_x, *self.popt)

            pred_return_y = super().resample_returns(
                returns=pred_monthly_return_y, m=days_in_month - 1
            )
            dev = pred_return_y - return_y
            self.deviation.append(dev)

            # Y is over-valued by a factor
            if return_y >= self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 0
                    # ):
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)

            # Y is undervalued by a factor
            elif return_y < self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 1
                    # ):
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.buy(row, sizing=False)

        else:
            self.hold(row)
        self.calculate_equity(row)


class GSP_NSH(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        first_trading_day, today, days_in_month, days_past = self.get_key_dates(row)
        if days_past > 0:
            return_x = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[0]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[0]}_close"]
                - 1,
                m=days_past,
            )

            return_y = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[1]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[1]}_close"]
                - 1,
                m=days_past,
            )

            monthly_return_x = super().resample_returns(
                returns=return_x, n=days_in_month - 1
            )
            pred_monthly_return_y = self.func(monthly_return_x, *self.popt)

            pred_return_y = super().resample_returns(
                returns=pred_monthly_return_y, m=days_in_month - 1
            )
            dev = pred_return_y - return_y
            self.deviation.append(dev)

            # Y is over-valued by a factor
            if return_y >= self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or ["state_5"] == 1 or row["state_200"] == 0
                    # ):
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)

            # Y is undervalued by a factor
            elif return_y < self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or row["state_20"] == 0 or row["state_200"] == 1
                    # ):
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.buy(row, sizing=False)

        else:
            self.hold(row)
        self.calculate_equity(row)


class TNI_ITQ(CorrelatedPairStrategy):
    def trade(self, row: pd.Series) -> None:
        first_trading_day, today, days_in_month, days_past = self.get_key_dates(row)
        if days_past > 0:
            return_x = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[0]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[0]}_close"]
                - 1,
                m=days_past,
            )

            return_y = super().resample_returns(
                returns=self.df.loc[today, f"{self.pair[1]}_close"]
                / self.df.loc[first_trading_day, f"{self.pair[1]}_close"]
                - 1,
                m=days_past,
            )

            monthly_return_x = super().resample_returns(
                returns=return_x, n=days_in_month - 1
            )
            pred_monthly_return_y = self.func(monthly_return_x, *self.popt)

            pred_return_y = super().resample_returns(
                returns=pred_monthly_return_y, m=days_in_month - 1
            )
            dev = pred_return_y - return_y
            self.deviation.append(dev)

            # Y is over-valued by a factor
            if return_y >= self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or row["state_5"] == 0 or row["state_200"] == 1
                    # ):
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)
                else:
                    if return_x > 0:
                        self.hold(row)
                    else:
                        self.sell(row)

            # Y is undervalued by a factor
            elif return_y < self.multiplier * pred_return_y:
                if return_y > 0:  # and (
                    # row["state_3"] == 0 or row["state_20"] == 1 or row["state_200"] == 0
                    # ):
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.hold(row)
                else:
                    if return_x > 0:
                        self.buy(row, sizing=False)
                    else:
                        self.buy(row, sizing=False)

        else:
            self.hold(row)
        self.calculate_equity(row)


def get_prices(pair: List) -> pd.DataFrame:
    """
    Get daily prices for two assets after the first 500 days only (2 years-ish).
    DONT TRADE the first 2 years after IPO because of extreme fluctuations.

    """
    freq = "daily"
    hp = HistoricalPrice()
    price_0 = hp.get_asset_price(pair[0], freq)
    price_1 = hp.get_asset_price(pair[1], freq)
    prices = pd.concat(
        [price_0[["close", "open"]], price_1[["close", "open"]]], axis=1
    ).dropna()
    prices.columns = [
        f"{pair[0]}_close",
        f"{pair[0]}_open",
        f"{pair[1]}_close",
        f"{pair[1]}_open",
    ]
    prices["close"] = price_1["close"]
    return prices.iloc[500:]


def get_states(pair: List) -> pd.DataFrame:
    """
    Get states based on target ticker.
    """
    hp = HistoricalPrice()
    price = hp.get_asset_price(pair[1], "daily")

    for lag in [3, 5, 20, 200]:
        clustered_df = cluster(pair[1], lag)
        price[f"state_{lag}"] = clustered_df["state"]
    return price.loc[:, [c for c in price.columns if "state" in c]]


def main():
    prices = get_prices(args.pair)
    states = get_states(args.pair)
    prices = pd.merge(states, prices, right_index=True, left_index=True)
    if not args.backtest:
        try:
            ticker_class = Ticker()
            for p in args.pair:
                if args.manual_price:
                    date = datetime.now().strftime("%m/%d/%Y")
                    price = float(input("price: "))
                else:
                    date, price = ticker_class.get_live_price(p)
                prices.loc[date, f"{p}_close"] = price
            prices.loc[date, "close"] = prices.loc[date, f"{p}_close"]
        except IndexError:
            message = f"No live price obtained: {args.pair}"
            slack.send_message(message)
            return

    prices["date"] = prices.index
    prices["week"] = prices.apply(lambda row: row["date"].isocalendar()[1], axis=1)
    if "_".join(args.pair) in globals():
        class_object = globals()["_".join(args.pair)]
    else:
        class_object = globals()["test"]
    c = class_object(**vars(args))
    c.execute(prices)


# MONTHLY, require average annual = 35% to trade
# python3 -m app.backtest.correlated_pair.correlated_pair --pair PLP DRH --max_dev 0.01 --multiplier 6 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 1.1961

# python3 -m app.backtest.correlated_pair.correlated_pair --pair PDR MBS --max_dev 0.02 --multiplier 5.5 --degree 2 --initial_capital 3000000 --max_portion 1 --backtest
# 0.3217 DONT TRADE

# python3 -m app.backtest.correlated_pair.correlated_pair --pair HAP EVG --max_dev 0.01 --multiplier 6 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.5906

# python3 -m app.backtest.correlated_pair.correlated_pair --pair GSP NSH --max_dev 0.01 --multiplier 4.5 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.7209

# python3 -m app.backtest.correlated_pair.correlated_pair --pair TNI ITQ --max_dev 0.01 --multiplier 2.5 --degree 2 --initial_capital 3000000 --max_portion 0.1 --backtest
# 1.1335


# WEEKLY, require average annual = 35% to trade
# python3 -m app.backtest.correlated_pair.correlated_pair --pair MBS BSI --max_dev 0.1 --multiplier 6 --degree 2 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.5322

# python3 -m app.backtest.correlated_pair.correlated_pair --pair CTS FTS --max_dev 0.25 --multiplier 10 --degree 2 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.6679

# python3 -m app.backtest.correlated_pair.correlated_pair --pair VGS TLH --max_dev 0.1 --multiplier 7 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.4165

# python3 -m app.backtest.correlated_pair.correlated_pair --pair VCI CTS --max_dev 0.1 --multiplier 5 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.6636

# python3 -m app.backtest.correlated_pair.correlated_pair --pair CTG HDB --max_dev 0.05 --multiplier 12 --degree 2 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.3278

# python3 -m app.backtest.correlated_pair.correlated_pair --pair MBB VND --max_dev 0.05 --multiplier 8 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.3736

# python3 -m app.backtest.correlated_pair.correlated_pair --pair STB LPB --max_dev 0.05 --multiplier 8 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.3051

# python3 -m app.backtest.correlated_pair.correlated_pair --pair VCG DIG --max_dev 0.05 --multiplier 15 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.3162

# python3 -m app.backtest.correlated_pair.correlated_pair --pair PLX PVS --max_dev 0.05 --multiplier 15 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.2996 < 0.3 DONT TRADE

# python3 -m app.backtest.correlated_pair.correlated_pair --pair VCI FTS --max_dev 0.05 --multiplier 13 --degree 1 --initial_capital 3000000 --max_portion 0.1 --backtest
# 0.6936


# WITH STATE INFORMATION, + WITH SIZING
# MONTHLY
# 1.1961 --> 1.2623
# 0.3868 --> 0.4258
# 0.5906 --> 0.5906
# 0.7209 --> 0.7209
# 1.1335 --> 1.1335

# WEEKLY
# 0.5322 --> 0.5461
# 0.6679 --> 0.7
# 0.4165 --> 0.4253
# 0.6636 --> 1.0929
# 0.3278 -> 0.3883
# 0.3736 -> 0.4723
# 0.3051 -> 0.3229
# 0.3162 -> 0.3295
# 0.6936 -> 0.8672

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs="+", default=[], required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--max_portion", type=float, required=True)
    parser.add_argument("--max_dev", type=float, required=True)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--initial_capital", type=float, required=True)
    parser.add_argument("--manual_price", action="store_true")
    parser.add_argument(
        "--degree",
        type=int,
        required=True,
        help="reduce prediction noise by fitting polynomial function",
    )

    args = parser.parse_args()
    main()
