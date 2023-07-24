from enum import Enum

from portfolio_analyzer_tool.constants import KEY_METRICS, SYMBOL, PERIOD, PE_RATIO, FREE_CASH_FLOW_YIELD, \
    DEBT_TO_EQUITY, DIVIDEND_YIELD, INTEREST_COVERAGE, PAYOUT_RATIO, DATE, INCOME_STATEMENT, REVENUE, GROSS_PROFIT, \
    GROSS_PROFIT_RATIO, OPERATING_INCOME, OPERATING_INCOME_RATIO, NET_INCOME, NET_INCOME_RATIO, EPS_DILUTED, \
    WEIGHTED_AVERAGE_SHARES_OUTSTANDING_DILUTED


class Datasets(Enum):
    KEY_METRICS = KEY_METRICS
    INCOME_STATEMENT = INCOME_STATEMENT


datasets_to_metrics_list_dict = {Datasets.KEY_METRICS: [SYMBOL, DATE, PERIOD, PE_RATIO, FREE_CASH_FLOW_YIELD,
                                                        DEBT_TO_EQUITY, DIVIDEND_YIELD, INTEREST_COVERAGE,
                                                        PAYOUT_RATIO],
                                 Datasets.INCOME_STATEMENT: [SYMBOL, DATE, PERIOD, REVENUE, GROSS_PROFIT,
                                                             GROSS_PROFIT_RATIO, OPERATING_INCOME,
                                                             OPERATING_INCOME_RATIO, NET_INCOME, NET_INCOME_RATIO,
                                                             EPS_DILUTED, WEIGHTED_AVERAGE_SHARES_OUTSTANDING_DILUTED]}
