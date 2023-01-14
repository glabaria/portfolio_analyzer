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

    def __init__(self, input_portfolio=None, save_file_path=None, benchmark_ticker_list=('dia', 'spy', 'qqq'),
                 benchmark_startdate_list=None, sliding_corr=None, sharp_ratio=None):
        self.transaction_csv_path = None
        self.portfolio_ticker_list = None
        self.portfolio_shares_list = None
        # input portfolio can either be a path or a list of tickers
        if os.path.exists(input_portfolio):
            self.transaction_csv_path = input_portfolio
        else:
            # regex \S+ \d+,
            x = input_portfolio.split(",")
            self.portfolio_ticker_list = [y[0] for y in x.split(" ")]
            self.portfolio_ticker_list = [int(y[1]) for y in x.split(" ")]
        self.save_file_path = save_file_path
        self.benchmark_ticker_list = benchmark_ticker_list
        self.benchmark_startdate_list = benchmark_startdate_list
        self.sliding_corr = sliding_corr
        self.shape_ratio = sharp_ratio
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

    def plot_sharpe_ratio(self, portfolio_sharpe_ratio_df, benchmark_sharpe_ratio_df):

        df = portfolio_sharpe_ratio_df[['sharpe_ratio']].join(benchmark_sharpe_ratio_df['sharpe_ratio'])
        df.rename(columns={'sharpe_ratio': 'Portfolio'}, inplace=True)
        df = pd.melt(df.reset_index(), id_vars='index').rename(columns={'index': 'Year', 'variable': 'Ticker',
                                                                        'value': 'Sharpe Ratio'})

        plt.figure(figsize=(10, 7))
        ax = sns.barplot(data=df, x='Year', y='Sharpe Ratio', hue='Ticker')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.tight_layout()
        plt.title('Annualized Sharpe Ratio')
        plt.savefig(os.path.join(self.save_file_path, 'sharpe_ratio.png'))

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

    @staticmethod
    def calculate_sharpe_ratio(df, daily_return_column_name='daily_return_pct'):
        """
        Function to calculate the sharpe_ratio := (Portfolio return - risk free return) / Std(Portfolio return).  We
        calculate the monthly Sharpe ratio by
            S_m := < Re_m > / std(Re_m)
        where < Re_m > is the average monthly excess return of the portfolio, and std(Re_m) is the std of the monthly
        excess return of the portfolio.  Here, the monthly excess return for the i-th month is given by
        Re_mi = Rp_mi - Rrf_mi where Rp_mi is the cumulative portfolio return for the i-th month and Rrf_mi is the
        monthly risk free return (the 13 week US Treasury Bill is used for the monthly risk free return).  The Sharpe
        ratio reported here is the annualized Sharpe ratio which is given by
            S_a := sqrt(12) * S_m

        :param df:  Dataframe containing the portfolio daily returns
        :param daily_return_column_name:
        :return:
        """

        three_month_tbill_df = yf.Ticker('^irx').history(start=df.Date.min())[['Close']]\
            .rename(columns={'Close': 'yield'})

        ticker_list = None
        daily_excess_return_column_name = ['daily_excess_return']
        cumulative_return_pct_column_name = ['cumulative_return_pct']
        sharpe_ratio_column_name = ['sharpe_ratio']
        # If Dataframe is multi-index, turn the three-month T-bill Dataframe and other appropriate columns to
        # multi-index
        if type(df.columns.values[0]) == tuple:
            three_month_tbill_df.columns = pd.MultiIndex.from_tuples([('yield', '^irx')])
            ticker_list = [y for x, y in df.columns.values if x == 'cumulative_return_pct']
            daily_excess_return_column_name = [(daily_excess_return_column_name[0], x) for x in ticker_list]
            cumulative_return_pct_column_name = [(cumulative_return_pct_column_name[0], x) for x in ticker_list]
            sharpe_ratio_column_name = [(sharpe_ratio_column_name[0], x) for x in ticker_list]

        # get monthly returns
        df = df.set_index('Date')
        df = df.join(three_month_tbill_df, how='inner')

        df[daily_excess_return_column_name] = df[[daily_return_column_name]] - df['yield'].values.reshape(-1, 1) / 365
        df_cum_return_month = df[daily_excess_return_column_name]\
            .groupby([lambda x: (x.year, x.month)]).apply(lambda x: (np.nanprod(1 + x / 100, axis=0) - 1) * 100)
        df_cum_return_month = pd.DataFrame(df_cum_return_month.to_list(), columns=cumulative_return_pct_column_name,
                                           index=df_cum_return_month.index)

        # get std of monthly returns for each year
        df_cum_return_std = df_cum_return_month.groupby([lambda x: x[0]])\
            .apply(lambda x: np.std(x, axis=0))[cumulative_return_pct_column_name]\
            .rename(columns={'cumulative_return_pct': 'monthly_return_std'} if ticker_list is None
                    else {x: ('monthly_return_std', x[1]) for x in cumulative_return_pct_column_name})

        # get average monthly returns for each year
        df_cum_return_month = df_cum_return_month.groupby(lambda x: x[0]).mean()\
            .rename(columns={'cumulative_return_pct': 'avg_cumulative_monthly_return'} if ticker_list is None
                    else {x: ('avg_cumulative_monthly_return', x[1]) for x in cumulative_return_pct_column_name})

        df_sharpe = df_cum_return_month.join(df_cum_return_std, how='inner')
        if ticker_list is not None:
            df_sharpe.columns = pd.MultiIndex.from_tuples(df_sharpe.columns.values)
        df_sharpe[sharpe_ratio_column_name[0] if ticker_list is None else sharpe_ratio_column_name] = \
            df_sharpe['avg_cumulative_monthly_return'] / df_sharpe['monthly_return_std'] * np.sqrt(12)

        return df_sharpe

    @staticmethod
    def calculate_sliding_correlation(x, y, window_length=10):
        if len(x) != len(y):
            raise ValueError(f'Inputs x and y must have the same length.')
        if window_length >= len(x) or window_length >= len(y):
            raise ValueError(f'Inputs window must be at most the length of x and y.')

        n_windows = len(x) - window_length + 1
        sliding_corr = np.zeros(n_windows)
        for window_ind in range(n_windows):
            meanx = np.mean(x[window_ind:window_length - 1 + window_ind])
            meany = np.mean(y[window_ind:window_length - 1 + window_ind])
            numerator = np.sum((x[window_ind:window_length - 1 + window_ind] - meanx) *
                               (y[window_ind:window_length - 1 + window_ind] - meany))
            denominator = np.sum((x[window_ind:window_length - 1 + window_ind] - meanx) ** 2)
            denominator *= np.sum((y[window_ind:window_length - 1 + window_ind] - meany) ** 2)
            denominator = np.sqrt(denominator)
            sliding_corr[window_ind] = numerator / denominator

        return sliding_corr

    def plot_sliding_correlation(self, df, window_length=10):

        plt.figure(figsize=(14, 8))

        for ticker in self.benchmark_ticker_list:
            plt.plot(df[ticker.upper()].values, label=ticker.upper(), linewidth=2)
        plt.legend(fontsize=14)
        dates = df.Date.dt.strftime('%Y-%m-%d').values
        plt.xticks(np.arange(0, len(dates), 5), dates[::5], rotation=90, fontsize=8)
        plt.yticks(fontsize=12)
        plt.xlabel(f'Date (each point representing window length {window_length} days)', fontsize=14)
        plt.ylabel('Correlation', fontsize=14)
        plt.title(f'Correlation to Benchmark (Window Length {window_length})', fontsize=14)
        os.makedirs(self.save_file_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_file_path, f'correlation_to_benchmark_window_{window_length}'))

    def plot_return_pct_helper(self, portfolio_df, benchmark_df, benchmark_ticker_list):
        """
        Wrapper function for plotting the cumulative returns of the portfolio vs benchmark for each days ago interval

        :param portfolio_df:
        :param benchmark_df:
        :param benchmark_ticker_list:
        :return:
        """
        # For each period, find the cumulative return percentage and plot
        base_title = 'Portfolio performance since '
        base_file_name = 'portfolio_performance_'
        file_name_list = [f"{x}" if "/" not in x else x.replace("/", "-") for x in self.benchmark_startdate_list]
        days_ago_list = [x if x != "inception" else -np.inf for x in self.benchmark_startdate_list]
        min_date, max_date = portfolio_df.Date.values[0], portfolio_df.Date.values[-1]
        for days_ago, file_name in zip(days_ago_list, file_name_list):
            if days_ago != -np.inf:
                if days_ago == "ytd":
                    date_cutoff = pd.to_datetime(f'01/01/{datetime.date.today().year}', format='%m/%d/%Y')
                elif "/" in days_ago:
                    date_cutoff = pd.to_datetime(days_ago, format='%m/%d/%Y')
                else:
                    date_cutoff = max_date - pd.Timedelta(days=int(days_ago))

                curr_portfolio_df = self.calculate_cumulative_returns_from_date(portfolio_df, date_cutoff)
                curr_benchmark_df = \
                    self.calculate_cumulative_returns_from_date(benchmark_df, date_cutoff,
                                                                cum_return_column_name=[('cumulative_return_pct', x)
                                                                                        for x in benchmark_ticker_list])
                self.plot_return_pct(curr_portfolio_df, curr_benchmark_df, title=base_title,
                                     file_name=base_file_name + file_name)
            else:
                self.plot_return_pct(portfolio_df, benchmark_df, file_name=base_file_name + file_name)

    def run(self):
        transaction_df, portfolio_value_df = self.gather_data()
        portfolio_df = self.calculate_daily_returns(transaction_df, portfolio_value_df)
        dates = portfolio_df.Date.dt.strftime('%Y-%m-%d').values
        benchmark_df, benchmark_ticker_list = \
            self.calculate_daily_return_from_ticker(self.benchmark_ticker_list, start_date=dates[0], end_date=dates[-1])
        benchmark_df.reset_index(inplace=True)
        portfolio_df, benchmark_df = self.ensure_equal_dates(portfolio_df, benchmark_df)

        # Plot the cumulative return percentage for each days ago interval
        self.plot_return_pct_helper(portfolio_df, benchmark_df, benchmark_ticker_list)

        # Calculate Sharpe ratio per year
        if self.shape_ratio:
            portfolio_sharpe_ratio_df = self.calculate_sharpe_ratio(portfolio_df)
            benchmark_sharpe_ratio_df = self.calculate_sharpe_ratio(benchmark_df)
            self.plot_sharpe_ratio(portfolio_sharpe_ratio_df, benchmark_sharpe_ratio_df)

        # Calculate portfolio correlations to each of the benchmark tickers
        if self.sliding_corr is not None:
            correlation_window = self.sliding_corr
            correlation_array = np.zeros((len(portfolio_df) - correlation_window + 1, len(self.benchmark_ticker_list)))
            for ind, benchmark_ticker in enumerate(self.benchmark_ticker_list):
                correlation_array[:, ind] = \
                    self.calculate_sliding_correlation(portfolio_df['cumulative_return_pct'].values,
                                                       benchmark_df['cumulative_return_pct']
                                                       [benchmark_ticker.upper()].values,
                                                       correlation_window)
            correlation_df = pd.DataFrame(correlation_array, columns=[x.upper() for x in self.benchmark_ticker_list])
            correlation_df.insert(0, 'Date', portfolio_df.Date.values[:len(portfolio_df) - correlation_window + 1])
            self.plot_sliding_correlation(correlation_df, window_length=correlation_window)


def run_portfolio_analyzer():
    transaction_csv_path = '../transaction_data/'
    save_file_path = '../figures/'
    benchmark_ticker_list = ['dia', 'spy', 'qqq', 'vt']
    PortfolioAnalyzer(input_portfolio=transaction_csv_path, save_file_path=save_file_path,
                      benchmark_ticker_list=benchmark_ticker_list).run()


if __name__ == '__main__':
    run_portfolio_analyzer()
