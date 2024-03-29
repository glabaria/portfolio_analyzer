import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import certifi
import json
from urllib.request import urlopen
from typing import Optional, List
from collections import defaultdict

from portfolio_analyzer_tool.constants import INDEX_KEYS_LIST, DATE, SYMBOL, YEAR, REVENUE, COST_OF_REVENUE, \
    OPERATING_EXPENSES, TOTAL_ASSETS, TOTAL_CURRENT_LIABILITIES, GROSS_MARGIN, OPERATING_MARGIN, NET_MARGIN, \
    GROSS_PROFIT, OPERATING_INCOME, NET_INCOME, FREE_CASH_FLOW_ADJUSTED, FREE_CASH_FLOW_YIELD_ADJUSTED, \
    FREE_CASH_FLOW, STOCK_BASED_COMPENSATION, MARKET_CAPITALIZATION, PERIOD, SUPPORTED_BASE_TTM_METRICS_LIST, \
    SUPPORTED_TTM_METRICS_LIST, YEAR_PERIOD, FY, QUARTER, FREE_CASH_FLOW_ADJUSTED_PER_SHARE, FREE_CASH_FLOW_PER_SHARE, \
    NUMBER_OF_SHARES, STOCK_BASED_COMPENSATION_AS_PCT_OF_FCF, DIVIDEND_YIELD, METRIC_FORMAT_DICT, MARKET_CAP, \
    CURR_MARKET_CAP, MARKET_CAPITALIZATION_FIELD, CALENDAR_YEAR
from portfolio_analyzer_tool.enum_types import Datasets, datasets_to_metrics_list_dict


class Fundamentals:
    def __init__(self, ticker_list, key):
        self.ticker_info_df = None
        self.ticker_list = ticker_list
        self.key = key
        self.current_ticker_info_df = None

    @staticmethod
    def _consolidate_dates(df_list):
        for df in df_list:
            df.dropna(subset=CALENDAR_YEAR, inplace=True)
            df[YEAR] = pd.to_datetime(df[CALENDAR_YEAR]).dt.year.to_numpy()
            df[YEAR_PERIOD] = df[[CALENDAR_YEAR, PERIOD]].apply(lambda x: f"{x[0]}-{x[1]}", axis=1).values

    def gather_current_datasets(self, dataset_list: List[str]) -> None:
        """ Gathers metrics for the current day.  Returns DataFrame of symbol | metric1 | metric 2 | ... | metric n """
        ticker_info_df_list = []
        for ticker in self.ticker_list:
            curr_ticker_work_df_list = [self.gather_dataset(ticker, dataset) for dataset in dataset_list]
            curr_ticker_work_df_list = [df.set_index([SYMBOL]) for df in curr_ticker_work_df_list]
            curr_ticker_work_df = curr_ticker_work_df_list[0].join(curr_ticker_work_df_list[1:], how="outer")
            ticker_info_df_list.append(curr_ticker_work_df)
        self.current_ticker_info_df = pd.concat(ticker_info_df_list, axis=0)

    def gather_all_datasets(self, dataset_list: Optional[List[str]] = None, period: Optional[str] = None) -> None:
        """
        Gathers fundamental metrics for each ticker in self.ticker_list

        :return:
        """
        is_enterprise_values_requested = False
        if dataset_list is None:
            dataset_list = [dataset.value for dataset in Datasets if dataset not in
                            [Datasets.ENTERPRISE_VALUES, Datasets.MARKET_CAPITALIZATION]]
        if Datasets.ENTERPRISE_VALUES in Datasets:
            is_enterprise_values_requested = True

        period = "fy" if period is None else period

        enterprise_value_df = None
        ticker_info_df_list = []
        for ticker in self.ticker_list:
            if is_enterprise_values_requested:
                enterprise_value_df = self.gather_dataset(ticker, Datasets.ENTERPRISE_VALUES.value, period=period)

            curr_ticker_work_df_list = [self.gather_dataset(ticker, dataset, period=period) for dataset in dataset_list]
            if enterprise_value_df is not None:
                # TODO: is there a better way to do this?  What happens if the DATE does not match up?
                curr_ticker_work_df_list[0] = pd.merge(curr_ticker_work_df_list[0],
                                                       enterprise_value_df[[x for x in datasets_to_metrics_list_dict
                                                       [Datasets.ENTERPRISE_VALUES] if x != SYMBOL]], on=[DATE])
            curr_ticker_work_df_list = [df.set_index([DATE, SYMBOL, PERIOD]) for df in curr_ticker_work_df_list]

            # drop DATE column from all but one dataset
            # FIXME: make this more general
            curr_ticker_work_df_list = \
                [df if i <= 1 else df.drop([CALENDAR_YEAR], axis=1)
                 for i, df in enumerate(curr_ticker_work_df_list)]

            curr_ticker_work_df = curr_ticker_work_df_list[0].join(curr_ticker_work_df_list[1:], how="outer")
            curr_ticker_work_df.reset_index(inplace=True)
            self._consolidate_dates([curr_ticker_work_df])
            curr_ticker_work_df.set_index(INDEX_KEYS_LIST, inplace=True)
            ticker_info_df_list.append(curr_ticker_work_df)

        self.ticker_info_df = pd.concat(ticker_info_df_list, axis=0)

    def gather_dataset(self, ticker: str, dataset: str, period: Optional[str] = None, **kwargs) -> pd.DataFrame:
        kwargs_to_use = dict(period=period, **kwargs) if period is not None else kwargs if kwargs is not None else {}
        json_data = self.get_jsonparsed_data(dataset, ticker, self.key, **kwargs_to_use)
        work_ticker_df = pd.DataFrame.from_records(json_data)
        ticker_info_df = work_ticker_df[datasets_to_metrics_list_dict[Datasets(dataset)]]
        return ticker_info_df

    @staticmethod
    def _year_period_key(x, ind):
        if ind == 0:
            x = x.apply(lambda s: s.split("-")[0])
        else:
            d = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            x = x.apply(lambda s: d[s.split("-")[1]])
        return x

    def plot_fundamentals(self, field_list: List[str], save_results_path: str, period: str, ttm_flag: bool = False,
                          df_pct_years_ago: Optional[pd.DataFrame] = None, df_stats: Optional[pd.DataFrame] = None):

        def _default_annotation(x):
            return f"{x:.{METRIC_FORMAT_DICT.get(field, '1f')}}" if not np.isnan(x) else ""

        for symbol in self.ticker_list + ["all"]:
            os.makedirs(os.path.join(save_results_path, symbol), exist_ok=True)
            mask = self.ticker_info_df.index.get_level_values("symbol") == symbol if symbol != "all" \
                else np.ones(len(self.ticker_info_df), dtype=bool)
            work_df = self.ticker_info_df.loc[mask].reset_index()
            work_df = work_df.sort_values(by=[YEAR, PERIOD])
            for field in field_list:
                plt.figure(figsize=(14 if period == FY else 31, 8))
                if symbol != "all":
                    g = sns.barplot(data=work_df, x=YEAR_PERIOD, y=field, color="b")
                    g.axes.bar_label(g.axes.containers[-1],
                                     labels=map(lambda x: self.format_number(x, _default_annotation),
                                                g.axes.containers[-1].datavalues),
                                     label_type='edge', rotation=90)
                else:
                    g = sns.lineplot(data=work_df, x=YEAR_PERIOD, y=field, hue=SYMBOL, marker=".", linewidth=2,
                                     markersize=25)
                    g.set_xticks(work_df[YEAR_PERIOD].values, work_df[YEAR_PERIOD].values)
                plt.xticks(rotation=90)
                plt.title(f"{field}{'(TTM)' if ttm_flag and field in SUPPORTED_TTM_METRICS_LIST else ''}")

                # write statistics as text at the top of the plot
                if symbol != "all":
                    text_str = ""
                    if df_pct_years_ago is not None:
                        curr_stats = df_pct_years_ago.loc[symbol, ["years_ago", field]].values
                        text_str += "CAGR %:\n"
                        for row_ind in range(curr_stats.shape[0]):
                            if np.isnan(curr_stats[row_ind, 1]):
                                continue
                            text_str += f"{int(curr_stats[row_ind, 0])} years: {curr_stats[row_ind, 1]:.2f}%\n"
                    if df_stats is not None:
                        text_str += "\n"
                        stats_list = df_stats.loc[symbol].index.values
                        for stat in stats_list:
                            value = df_stats.loc[(symbol, stat), field]
                            text_str += f"{stat}: {self.format_number(value, _default_annotation)}\n"
                    if self.current_ticker_info_df is not None and field == FREE_CASH_FLOW_YIELD_ADJUSTED:
                        value = self.current_ticker_info_df.loc[symbol, field]
                        text_str += f"today: {self.format_number(value, _default_annotation)}"

                    plt.gcf().text(0.2, 0.85, text_str)
                plt.savefig(os.path.join(save_results_path, symbol,
                                         f"{field}{'_ttm' if ttm_flag and field in SUPPORTED_TTM_METRICS_LIST else ''}.png"),
                            dpi=300, bbox_inches="tight")
                plt.close()

    @staticmethod
    def get_jsonparsed_data(dataset_name: str, ticker: str, key: str,
                            base_url: str = "https://financialmodelingprep.com/api/v3",
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

    def convert_ratio_to_pct(self):
        self.ticker_info_df[DIVIDEND_YIELD] = self.ticker_info_df[DIVIDEND_YIELD] * 100

    def calculate_ttm(self):
        self.ticker_info_df = self.ticker_info_df.sort_values(by=YEAR_PERIOD)
        self.ticker_info_df[SUPPORTED_BASE_TTM_METRICS_LIST] = \
            self.ticker_info_df[SUPPORTED_BASE_TTM_METRICS_LIST].groupby(SYMBOL, as_index=False).rolling(4).sum()\
                .drop(columns=[SYMBOL])

    def calculate_roce(self):
        ebit = self.ticker_info_df[REVENUE] - self.ticker_info_df[COST_OF_REVENUE] - \
               self.ticker_info_df[OPERATING_EXPENSES]
        self.ticker_info_df["roce"] = \
            ebit / (self.ticker_info_df[TOTAL_ASSETS] - self.ticker_info_df[TOTAL_CURRENT_LIABILITIES]) * 100

    def calculate_margins(self):
        self.ticker_info_df[GROSS_MARGIN] = self.ticker_info_df[GROSS_PROFIT] / self.ticker_info_df[REVENUE] * 100
        self.ticker_info_df[OPERATING_MARGIN] = \
            self.ticker_info_df[OPERATING_INCOME] / self.ticker_info_df[REVENUE] * 100
        self.ticker_info_df[NET_MARGIN] = self.ticker_info_df[NET_INCOME] / self.ticker_info_df[REVENUE] * 100

    def calculate_adjusted_fcf(self):
        self.ticker_info_df[FREE_CASH_FLOW_ADJUSTED] = \
            self.ticker_info_df[FREE_CASH_FLOW] - self.ticker_info_df[STOCK_BASED_COMPENSATION]
        self.ticker_info_df[FREE_CASH_FLOW_YIELD_ADJUSTED] = \
            self.ticker_info_df[FREE_CASH_FLOW_ADJUSTED] / self.ticker_info_df[MARKET_CAPITALIZATION_FIELD] * 100

        # calculate per share
        self.ticker_info_df[FREE_CASH_FLOW_PER_SHARE] = \
            self.ticker_info_df[FREE_CASH_FLOW] / self.ticker_info_df[NUMBER_OF_SHARES]
        self.ticker_info_df[FREE_CASH_FLOW_ADJUSTED_PER_SHARE] = \
            self.ticker_info_df[FREE_CASH_FLOW_ADJUSTED] / self.ticker_info_df[NUMBER_OF_SHARES]

        # calculate SBC as pct of FCF
        self.ticker_info_df[STOCK_BASED_COMPENSATION_AS_PCT_OF_FCF] = \
            self.ticker_info_df[STOCK_BASED_COMPENSATION] / self.ticker_info_df[FREE_CASH_FLOW] * 100

        # calculate current free cash flow yield
        self.current_ticker_info_df = \
            pd.merge(self.current_ticker_info_df,
                     self.ticker_info_df.reset_index()[[SYMBOL, FREE_CASH_FLOW_ADJUSTED]]
                     .groupby(SYMBOL).tail(1),
                     on="symbol", how="left")
        self.current_ticker_info_df[FREE_CASH_FLOW_YIELD_ADJUSTED] = \
            self.current_ticker_info_df[FREE_CASH_FLOW_ADJUSTED] / self.current_ticker_info_df[MARKET_CAP] * 100
        del self.current_ticker_info_df[FREE_CASH_FLOW_ADJUSTED]
        self.current_ticker_info_df.set_index(SYMBOL, inplace=True)

    def calculate_pct_change_from_years_ago(self, data, years_ago, period, metrics_list):
        year_today = pd.Timestamp.today().year
        if year_today not in data[YEAR].unique():
            year_today -= 1  # happens if current FY has not been reported yet
        years_ago_year = year_today - years_ago
        today_quarter = \
            None if period == FY else data.loc[data[YEAR] == year_today].index.get_level_values(YEAR_PERIOD).values[-1].split("-")[-1]
        work_df0 = data.loc[data[YEAR] == years_ago_year]
        work_df1 = data.loc[data[YEAR] == year_today]
        if today_quarter is None:
            metrics_array0 = work_df0[metrics_list].values
            metrics_array1 = work_df1[metrics_list].values
        else:
            metrics_array0 = work_df0[work_df0.index.get_level_values(PERIOD) == today_quarter][metrics_list].values
            metrics_array1 = work_df1[work_df1.index.get_level_values(PERIOD) == today_quarter][metrics_list].values

        pct_change_array = ((metrics_array1 / metrics_array0) ** (1 / years_ago) - 1) * 100
        pct_change_array[np.isinf(pct_change_array)] = np.nan

        return pct_change_array.flatten()

    def calculate_stats(self, metrics_list, period):
        df_pct_years_ago_list = []
        df_stats_list = []
        for symbol in self.ticker_list:
            curr_df = self.ticker_info_df[self.ticker_info_df.index.get_level_values(SYMBOL) == symbol]
            pct_change_dict = defaultdict(list)
            stats_df = curr_df[metrics_list].median().to_frame("median")
            stats_df = stats_df.join(curr_df[metrics_list].quantile(q=0.05).to_frame("5th-percentile"))
            stats_df = stats_df.join(curr_df[metrics_list].quantile(q=0.95).to_frame("95th-percentile"))
            stats_df.columns = pd.MultiIndex.from_tuples([(symbol, x) for x in stats_df.columns.values])
            df_stats_list.append(stats_df.T)

            # calculate year ago calculations
            for years_ago in [20, 10, 5, 3, 1]:
                pct_change_array = self.calculate_pct_change_from_years_ago(curr_df, years_ago, period, metrics_list)
                if not len(pct_change_array):
                    for metric in metrics_list:
                        pct_change_dict[metric].append(np.nan)
                else:
                    for metric, pct_change in zip(metrics_list, pct_change_array):
                        pct_change_dict[metric].append(pct_change)
                pct_change_dict["years_ago"].append(years_ago)
                pct_change_dict[SYMBOL].append(symbol)
            df_pct_years_ago_list.append(pd.DataFrame(pct_change_dict))
        df_pct_years_ago = pd.concat(df_pct_years_ago_list, axis=0, ignore_index=True).set_index(SYMBOL)
        df_stats = pd.concat(df_stats_list, axis=0)
        return df_pct_years_ago, df_stats

    def calculate_metrics(self, metrics_list, period, ttm_flag=False):
        if ttm_flag:
            self.calculate_ttm()
        self.calculate_roce()
        self.calculate_margins()
        self.calculate_adjusted_fcf()
        df_pct_years_ago, df_stats = self.calculate_stats(metrics_list, period)
        self.convert_ratio_to_pct()
        return df_pct_years_ago, df_stats

    @staticmethod
    def format_number(number, default_function=None):
        if abs(number) >= 1e9:
            return f"{round(number / 1e9, 2)}B"
        elif abs(number) >= 1e6:
            return f"{round(number / 1e6, 2)}M"
        elif abs(number) >= 1e3:
            return f"{round(number / 1e3, 2)}K"
        else:
            return str(number) if default_function is None else default_function(number)


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
