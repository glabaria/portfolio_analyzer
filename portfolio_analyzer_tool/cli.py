import argparse
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
                             "mm/dd/yyyy to compare since date.")

    args = parser.parse_args()
    ticker_list = ['dia', 'spy', 'qqq', 'vt'] if args.tickers is None else args.tickers.split(",")
    pa_obj = PortfolioAnalyzer(transaction_csv_path=args.input_dir, save_file_path=args.output_dir,
                               benchmark_ticker_list=ticker_list)
    pa_obj.run()


if __name__ == "__main__":
    cli()
