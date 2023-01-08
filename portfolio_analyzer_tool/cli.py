import argparse
import numpy as np
from portfolio_analyzer_tool.portfolio_analyzer import PortfolioAnalyzer


def cli():
    parser = argparse.ArgumentParser(description="Portfolio Analyzer")
    parser.add_argument("-i", "--input_dir", type=str,
                        help="Path to directory which contains the 'chart.csv' and 'transactions_{year}.csv' files")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Path to directory to save figures")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Flag to benchmark portfolio")
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

    args = parser.parse_args()

    # get ticket list to compare against
    ticker_list = ['dia', 'spy', 'qqq', 'vt'] if args.tickers is None else args.tickers.split(",")

    # get days to compare
    from_da_list = [30, 90, 180, 365, 730, 'ytd', -np.inf] if args.from_da is None else args.from_da.split(",")

    pa_obj = PortfolioAnalyzer(transaction_csv_path=args.input_dir, save_file_path=args.output_dir,
                               benchmark_ticker_list=ticker_list, benchmark_startdate_list=from_da_list)
    pa_obj.run()


if __name__ == "__main__":
    cli()
