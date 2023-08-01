import argparse
import os
import numpy as np
from portfolio_analyzer_tool.portfolio_analyzer import PortfolioAnalyzer


def cli():
    parser = argparse.ArgumentParser(description="Portfolio Analyzer")
    parser.add_argument("-p", "--portfolio", type=str,
                        help="Path to directory which contains the 'chart.csv' and 'transactions_{year}.csv' files for"
                             "the portfolio;"
                             "or input a list of tickers followed by the share amount to form a portfolio: "
                             "e.g., 'aapl 100,msft 200' for a portfolio consisting of 100 of aapl shares and 200 "
                             "of msft shares.")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Path to directory to save results.  Defaults to current working directory.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Flag to benchmark portfolio", default=False)
    parser.add_argument("--tickers", type=str,
                        help="Ticker list to compare to separated by commas; e.g. 'spy,qqq'")
    parser.add_argument("--from_da", type=str,
                        help="Days ago to compare portfolio to tickers.  "
                             "Valid options include:\n"
                             "x where x is an integer to compare x days ago;\n"
                             "ytd to compare year to date;\n"
                             "inception to compare since inception;\n"
                             "mm/dd/yyyy to compare since date.\n"
                             "Can be combined with multiple days: e.g., '30,90,180,ytd,inception,01/01/2020'\n"
                             "for comparing 30, 90, 180 days ago, year-to-date, inception, and since the start of 2020."
                        )
    parser.add_argument("--sliding_corr", type=int, help="Number of days for the sliding correlation window")
    parser.add_argument("--sharpe_ratio", action="store_true", help="Flag to calculate the sharpe ratio per year")
    parser.add_argument("--fundamental", type=str, help="Outputs fundamental data specified by year or quarter."
                                                        "Fundamental data requested should be comma separated.  "
                                                        "For example, 'freeCashFlowYield,interestCoverage'")
    parser.add_argument("--period", type=str, help="Period for fundamental data.  Options are 'quarter' or 'fy'.")

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else os.getcwd()

    # get ticket list to compare against
    ticker_list = ['dia', 'spy', 'qqq', 'vt'] if args.tickers is None else args.tickers.split(",")

    # get days to compare
    from_da_list = [30, 90, 180, 365, 730, 'ytd', -np.inf] if args.from_da is None else args.from_da.split(",")

    pa_obj = PortfolioAnalyzer(input_portfolio=args.portfolio, save_file_path=output_dir,
                               benchmark_ticker_list=ticker_list, benchmark_startdate_list=from_da_list,
                               sharp_ratio=args.sharpe_ratio, sliding_corr=args.sliding_corr,
                               fundamental_data=args.fundamental, period=args.period, benchmark_flag=args.benchmark)
    pa_obj.run()


if __name__ == "__main__":
    cli()
