import yfinance as yf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

DEPOSIT_DESCRIPTION_LIST = ['ELECTRONIC NEW ACCOUNT FUNDING', 'CLIENT REQUESTED ELECTRONIC FUNDING RECEIPT (FUNDS NOW)']
IGNORE_LIST = ['INTRA-ACCOUNT TRANSFER']


class PortfolioAnalyzer:

    def __init__(self, transaction_csv_path=None, save_file_path=None, benchmark_ticker_list=('dia', 'spy', 'qqq')):
        self.transaction_csv_path = transaction_csv_path
        self.save_file_path = save_file_path
        self.benchmark_ticker_list = benchmark_ticker_list
        sns.set()

    def gather_data(self):
        """
        Helper function to to gather transaction and account balance csvs

        :return:
            Return tuple (transaction_df, portfolio_value_df)
        """

        # read transaction CSV data
        filename_list = os.listdir(self.transaction_csv_path)
        transaction_df = None
        for filename in filename_list:
            if filename == 'chart.csv':
                continue
            if transaction_df is None:
                transaction_df = pd.read_csv(os.path.join(self.transaction_csv_path, filename))
            else:
                transaction_df = transaction_df.append(pd.read_csv(os.path.join(self.transaction_csv_path, filename)))
        portfolio_value_df = pd.read_csv(os.path.join(self.transaction_csv_path, 'chart.csv'))

        return transaction_df, portfolio_value_df

    @staticmethod
    def calculate_daily_returns(transaction_df, portfolio_value_df):
        """
        Calculate daily returns from TD Ameritrade transaction and portfolio Dataframes

        :param transaction_df: Dataframe consisting of the user's transactions
        :param portfolio_value_df: Dataframe consisting of the portfolio's balence
        :return:
            Dataframe consisting of portfolio daily and cumulative returns
        """
        # get dates and amount of any account deposits
        transaction_df['IS_DEPOSIT'] = transaction_df['DESCRIPTION'].apply(lambda x: x in DEPOSIT_DESCRIPTION_LIST)

        portfolio_value_df = pd.merge(portfolio_value_df, transaction_df, how='left', left_on='Date', right_on='DATE')
        # delete last three rows, because they do not contain information
        portfolio_value_df = portfolio_value_df.iloc[:-3, :]
        portfolio_value_df.drop(columns=['DATE'], inplace=True)
        portfolio_value_df.loc[portfolio_value_df['IS_DEPOSIT'].isna(), 'IS_DEPOSIT'] = False

        # drop rows with intra-account transfer
        portfolio_value_df = portfolio_value_df.loc[~portfolio_value_df.DESCRIPTION.isin(IGNORE_LIST), :]

        # calculate daily return, taking into consideration deposits
        portfolio_value_df['Account value'] = \
            portfolio_value_df['Account value'].apply(lambda x: float(x.replace(',', '')))
        portfolio_value_df['ACCOUNT_VALUE_EX_DEPOSIT'] = portfolio_value_df['Account value']
        portfolio_value_df.loc[portfolio_value_df['IS_DEPOSIT'].values, 'ACCOUNT_VALUE_EX_DEPOSIT'] -= \
            portfolio_value_df.loc[portfolio_value_df['IS_DEPOSIT'].values, 'AMOUNT'].values
        portfolio_value_unique_date = \
            portfolio_value_df.loc[~portfolio_value_df.duplicated(['Date']),
                                   ['Date', 'Account value', 'ACCOUNT_VALUE_EX_DEPOSIT']]
        portfolio_value_unique_date_with_deposit = \
            portfolio_value_df.loc[portfolio_value_df.IS_DEPOSIT, ['Date', 'IS_DEPOSIT', 'Account value',
                                                                   'ACCOUNT_VALUE_EX_DEPOSIT']]
        portfolio_value_unique_date.loc[
            portfolio_value_unique_date.Date.isin(portfolio_value_unique_date_with_deposit.Date.values),
            ['Account value', 'ACCOUNT_VALUE_EX_DEPOSIT']] = \
            portfolio_value_unique_date_with_deposit[['Account value', 'ACCOUNT_VALUE_EX_DEPOSIT']].values
        portfolio_value_unique_date['daily_return_pct'] = \
            np.zeros_like(portfolio_value_unique_date.iloc[:, 1].values)
        portfolio_value_unique_date.loc[1:, 'daily_return_pct'] = \
            np.round((portfolio_value_unique_date.iloc[1:, -2].values /
                      portfolio_value_unique_date.iloc[:-1, -3].values - 1) * 100, 2)
        portfolio_value_unique_date['cumulative_return_pct'] = \
            (np.nancumprod(1 + portfolio_value_unique_date.daily_return_pct.values / 100) - 1) * 100

        portfolio_value_unique_date['Date'] = pd.to_datetime(portfolio_value_unique_date['Date'], format='%m/%d/%Y')

        return portfolio_value_unique_date

    @staticmethod
    def calculate_daily_return_from_ticker(ticker_list, start_date, end_date):
        """
        Calculate daily returns from a ticker list

        :param ticker_list: List of symbols to calculate returns for
        :param start_date: Start date of query
        :param end_date: End date of query
        :return: yfinance Dataframe of ticker data
        """

        yf_df = yf.download(ticker_list, start=start_date, end=end_date, interval='1d')
        ind_array = np.array([(i, x[-1]) for i, x in enumerate(yf_df.columns.values) if x[0] == 'Adj Close'])
        ind_array, ticker_list = np.array([int(x[0]) for x in ind_array]), [x[1] for x in ind_array]

        # calculate daily return
        col_array = [('daily_return_pct', x) for x in ticker_list]
        n_tickers = len(ticker_list)
        yf_df[col_array] = np.zeros_like(yf_df.iloc[:, ind_array].values)
        yf_df.iloc[1:, -n_tickers:] = \
            np.round((yf_df.iloc[1:, ind_array].values / yf_df.iloc[:-1, :n_tickers].values - 1) * 100, 2)

        # calculate cumulative return
        col_array = [('cumulative_return_pct', x) for x in ticker_list]
        yf_df[col_array] = (np.nancumprod(1 + yf_df.iloc[:, -n_tickers:].values / 100, axis=0) - 1) * 100

        return yf_df, ticker_list

    def plot_return_pct(self, portfolio_df, benchmark_df, title='Portfolio performance since inception',
                        file_name='portfolio_performance'):
        """
        Function to plot portfolio return pct

        :param portfolio_df:
        :param benchmark_df:
        :param title:
        :param file_name:
        :return:
        """

        plt.figure(figsize=(14, 8))
        plt.plot(portfolio_df.cumulative_return_pct.values, label='Portfolio', linewidth=2)

        benchmark_ticker_list = [y for x, y in benchmark_df.columns.values if x == 'cumulative_return_pct']

        for ticker in benchmark_ticker_list:
            plt.plot(benchmark_df[('cumulative_return_pct', ticker)].values, label=ticker.upper(), linewidth=2)
        plt.legend(fontsize=14)
        dates = portfolio_df.Date.dt.strftime('%Y-%m-%d').values
        plt.xticks(np.arange(0, len(dates), 5), dates[::5], rotation=90, fontsize=10)
        plt.yticks(fontsize=12)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Cumulative Return %', fontsize=14)
        plt.title(title, fontsize=14)
        os.makedirs(self.save_file_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_file_path, file_name))

    @staticmethod
    def ensure_equal_dates(df1, df2):
        """
        Utility function to ensure that df1 and df2 have the same dates.  Dates that are not in the intersection of
        df1.Date and df2.Date are discarded.

        :param df1: First Dataframe (note: must contain a column called "Date")
        :param df2: Second Dataframe (note: must contain a column called "Date")
        :return: Tuple (df1, df2)
        """
        date1 = set(list(df1.Date.values))
        date2 = set(list(df2.Date.values))
        date = date1.intersection(date2)
        df1 = df1.loc[df1.Date.isin(date), :]
        df2 = df2.loc[df2.Date.isin(date), :]
        return df1, df2

    @staticmethod
    def calculate_cumulative_returns_from_date(df, start_date, daily_return_column_name='daily_return_pct',
                                               cum_return_column_name=None):
        """
        This function calculates the cumulative return % from the daily return %.

        :param df: Dataframe containing the daily_return_column_name
        :param start_date: Pandas datetime to start calculating the cumulative daily return from
        :param daily_return_column_name: Name of the column that contains the daily return %
        :param cum_return_column_name: Name of the column to store the cumulative return calculated (can also be tuple
            for multi-indexed columns)
        :return:
            Copy of dataframe df that contains the cum_return_column_name of cumulative returns starting from
            start_date.  Note that the dates in this Dataframe are truncated to only include start_date onwards
        """

        df = df.copy()
        df = df.loc[df.Date >= start_date]
        if cum_return_column_name is None:
            cum_return_column_name = 'cumulative_return_pct'
        df[cum_return_column_name] = (np.nancumprod(1 + df[daily_return_column_name].values / 100, axis=0) - 1) * 100
        return df

    def run(self):
        transaction_df, portfolio_value_df = self.gather_data()
        portfolio_df = self.calculate_daily_returns(transaction_df, portfolio_value_df)
        dates = portfolio_df.Date.dt.strftime('%Y-%m-%d').values
        benchmark_df, ticker_list = \
            self.calculate_daily_return_from_ticker(self.benchmark_ticker_list, start_date=dates[0], end_date=dates[-1])
        benchmark_df.reset_index(inplace=True)
        portfolio_df, benchmark_df = self.ensure_equal_dates(portfolio_df, benchmark_df)

        # For each period, find the cumulative return percentage
        base_title = 'Portfolio performance since '
        base_file_name = 'portfolio_performance_'
        days_ago_list = [30, 90, 180, 365, 'ytd', -np.inf]
        title_appendix_list = ['last month', 'three months ago', 'six months ago', 'one year ago', 'YTD', 'inception']
        file_name_list = ['30_days', '90_days', '180_days', '365_days', 'ytd', 'inception']
        min_date, max_date = portfolio_df.Date.values[0], portfolio_df.Date.values[-1]
        for days_ago, title_appendix, file_name in zip(days_ago_list, title_appendix_list, file_name_list):
            if days_ago != -np.inf:
                date_cutoff = max_date - pd.Timedelta(days=days_ago) if days_ago != 'ytd' \
                    else pd.to_datetime(f'01/01/{datetime.date.today().year}', format='%m/%d/%Y')
                curr_portfolio_df = self.calculate_cumulative_returns_from_date(portfolio_df, date_cutoff)
                curr_benchmark_df = \
                    self.calculate_cumulative_returns_from_date(benchmark_df, date_cutoff,
                                                                cum_return_column_name=[('cumulative_return_pct', x)
                                                                                        for x in ticker_list])
                self.plot_return_pct(curr_portfolio_df, curr_benchmark_df, title=base_title + title_appendix,
                                     file_name=base_file_name + file_name)
            else:
                self.plot_return_pct(portfolio_df, benchmark_df, file_name=base_file_name + file_name)


def run_portfolio_analyzer():
    transaction_csv_path = './transaction_data/'
    save_file_path = './figures/'
    benchmark_ticker_list = ['dia', 'spy', 'qqq']
    PortfolioAnalyzer(transaction_csv_path=transaction_csv_path, save_file_path=save_file_path,
                      benchmark_ticker_list=benchmark_ticker_list).run()


if __name__ == '__main__':
    run_portfolio_analyzer()
