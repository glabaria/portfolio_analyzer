from enum import Enum

from portfolio_analyzer_tool.constants import KEY_METRICS, SYMBOL, PE_RATIO, FREE_CASH_FLOW_YIELD, \
    DEBT_TO_EQUITY, DIVIDEND_YIELD, INTEREST_COVERAGE, PAYOUT_RATIO, DATE, INCOME_STATEMENT, REVENUE, GROSS_PROFIT, \
    GROSS_PROFIT_RATIO, OPERATING_INCOME, OPERATING_INCOME_RATIO, NET_INCOME, NET_INCOME_RATIO, EPS_DILUTED, \
    WEIGHTED_AVERAGE_SHARES_OUTSTANDING_DILUTED, COST_OF_REVENUE, OPERATING_EXPENSES, BALANCE_SHEET_STATEMENT, \
    TOTAL_ASSETS, TOTAL_CURRENT_LIABILITIES, CASH_FLOW_STATEMENT, FREE_CASH_FLOW, STOCK_BASED_COMPENSATION, \
    ENTERPRISE_VALUES, MARKET_CAPITALIZATION, INDEX_KEYS_LIST, PERIOD, NUMBER_OF_SHARES, MARKET_CAPITALIZATION_FIELD, \
    MARKET_CAP, CALENDAR_YEAR


class Datasets(Enum):
    KEY_METRICS = KEY_METRICS
    INCOME_STATEMENT = INCOME_STATEMENT
    BALANCE_SHEET_STATEMENT = BALANCE_SHEET_STATEMENT
    CASH_FLOW_STATEMENT = CASH_FLOW_STATEMENT
    ENTERPRISE_VALUES = ENTERPRISE_VALUES
    MARKET_CAPITALIZATION = MARKET_CAPITALIZATION


datasets_to_metrics_list_dict = {Datasets.KEY_METRICS: [PE_RATIO, FREE_CASH_FLOW_YIELD,
                                                        DEBT_TO_EQUITY, DIVIDEND_YIELD, INTEREST_COVERAGE,
                                                        PAYOUT_RATIO, DATE, SYMBOL, PERIOD],
                                 Datasets.INCOME_STATEMENT: [REVENUE, GROSS_PROFIT,
                                                             GROSS_PROFIT_RATIO, OPERATING_INCOME,
                                                             OPERATING_INCOME_RATIO, NET_INCOME, NET_INCOME_RATIO,
                                                             EPS_DILUTED, WEIGHTED_AVERAGE_SHARES_OUTSTANDING_DILUTED,
                                                             COST_OF_REVENUE, OPERATING_EXPENSES, DATE, SYMBOL, PERIOD,
                                                             CALENDAR_YEAR],
                                 Datasets.BALANCE_SHEET_STATEMENT: [TOTAL_ASSETS, TOTAL_CURRENT_LIABILITIES, DATE,
                                                                    SYMBOL, PERIOD, CALENDAR_YEAR],
                                 Datasets.CASH_FLOW_STATEMENT: [FREE_CASH_FLOW, STOCK_BASED_COMPENSATION, DATE, SYMBOL,
                                                                PERIOD, CALENDAR_YEAR],
                                 Datasets.ENTERPRISE_VALUES: [MARKET_CAPITALIZATION_FIELD, NUMBER_OF_SHARES, SYMBOL,
                                                              DATE],
                                 Datasets.MARKET_CAPITALIZATION: [SYMBOL, MARKET_CAP]}
