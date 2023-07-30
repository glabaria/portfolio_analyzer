import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import certifi
import json
from urllib.request import urlopen
from typing import Optional, List

from portfolio_analyzer_tool.constants import INDEX_KEYS_LIST, DATE, SYMBOL, YEAR, REVENUE, COST_OF_REVENUE, \
    OPERATING_EXPENSES, TOTAL_ASSETS, TOTAL_CURRENT_LIABILITIES, GROSS_MARGIN, OPERATING_MARGIN, NET_MARGIN, \
    GROSS_PROFIT_RATIO, OPERATING_INCOME_RATIO, NET_INCOME_RATIO
from portfolio_analyzer_tool.enum_types import Datasets, datasets_to_metrics_list_dict


class Fundamentals:
    def __init__(self, ticker_list, key):
        self.ticker_info_df = None
        self.ticker_list = ticker_list
        self.key = key

    def _consolidate_dates(self):
        self.ticker_info_df[YEAR] = pd.to_datetime(self.ticker_info_df.reset_index()[DATE]).dt.year.to_numpy()

    def gather_all_datasets(self, dataset_list: Optional[List[str]] = None) -> None:
        """
        Gathers fundamental metrics for each ticker in self.ticker_list

        :return:
        """

        dataset_list = [dataset.value for dataset in Datasets] if dataset_list is None else dataset_list
        ticker_info_df_list = [self.gather_dataset(dataset).set_index(INDEX_KEYS_LIST) for dataset in dataset_list]
        self.ticker_info_df = ticker_info_df_list[0].join(ticker_info_df_list[1:], how="outer")
        self._consolidate_dates()

    def gather_dataset(self, dataset: str) -> pd.DataFrame:
        ticker_info_df = None
        for ticker in self.ticker_list:
            json_data = self.get_jsonparsed_data(dataset, ticker, self.key)
            work_ticker_df = pd.DataFrame.from_records(json_data)
            work_ticker_df = work_ticker_df[INDEX_KEYS_LIST + datasets_to_metrics_list_dict[Datasets(dataset)]]
            if ticker_info_df is None:
                ticker_info_df = work_ticker_df
            else:
                ticker_info_df = pd.concat([ticker_info_df, work_ticker_df], axis=0, ignore_index=True)
        return ticker_info_df

    def plot_fundamentals(self, field_list: List[str], save_results_path: str):
        for symbol in self.ticker_list + ["all"]:
            os.makedirs(os.path.join(save_results_path, symbol), exist_ok=True)
            mask = self.ticker_info_df.index.get_level_values("symbol") == symbol if symbol != "all" \
                else np.ones(len(self.ticker_info_df), dtype=bool)
            work_df = self.ticker_info_df.loc[mask].reset_index()
            for field in field_list:
                plt.figure(figsize=(14, 8))
                if symbol != "all":
                    sns.barplot(data=work_df, x=YEAR, y=field, color="b")
                else:
                    sns.barplot(data=work_df, x=YEAR, y=field, hue=SYMBOL)
                plt.xticks(rotation=90)
                plt.savefig(os.path.join(save_results_path, symbol, f"{field}.png"), dpi=300)
                plt.close()

    @staticmethod
    def get_jsonparsed_data(dataset_name: str, ticker: str, key: str,
                            base_url: str ="https://financialmodelingprep.com/api/v3",
                            **kwargs) -> dict:
        """
        Receive the content of from a url of the form f"{base_url}/{dataset_name}/{ticker}?apikey={key}".

        Parameters
        ----------
        dataset_name
        base_url : str
        ticker
        key
        **kwargs:

        Returns
        -------
        dict
        """
        url = f"{base_url}/{dataset_name}/{ticker}?apikey={key}"
        for key, value in kwargs.items():
            url += f"&{key}={value}"
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)

    def calculate_roce(self):
        ebit = self.ticker_info_df[REVENUE] - self.ticker_info_df[COST_OF_REVENUE] - \
               self.ticker_info_df[OPERATING_EXPENSES]
        self.ticker_info_df["roce"] = \
            ebit / (self.ticker_info_df[TOTAL_ASSETS] - self.ticker_info_df[TOTAL_CURRENT_LIABILITIES]) * 100

    def calculate_margins(self):
        self.ticker_info_df[GROSS_MARGIN] = (1 - self.ticker_info_df[GROSS_PROFIT_RATIO]) * 100
        self.ticker_info_df[OPERATING_MARGIN] = (1 - self.ticker_info_df[OPERATING_INCOME_RATIO]) * 100
        self.ticker_info_df[NET_MARGIN] = (1 - self.ticker_info_df[NET_INCOME_RATIO]) * 100

    def calculate_metrics(self):
        self.calculate_roce()
        self.calculate_margins()


def _debug():
    with open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_files', 'key.txt')))) \
            as f:
        key = f.readlines()[0]

    ticker_list = ["MSFT", "AAPL", "NVDA"]
    fundamentals = Fundamentals(ticker_list, key)
    fundamentals.gather_all_datasets()


if __name__ == "__main__":
    # entrypoint for debug
    _debug()

