# datasets
KEY_METRICS = "key-metrics"
INCOME_STATEMENT = "income-statement"
BALANCE_SHEET_STATEMENT = "balance-sheet-statement"
CASH_FLOW_STATEMENT = "cash-flow-statement"
ENTERPRISE_VALUES = "enterprise-values"

# index keys
DATE = "date"
SYMBOL = "symbol"
PERIOD = "period"
YEAR = "year"
YEAR_PERIOD = "year-period"
INDEX_KEYS_LIST = [YEAR_PERIOD, SYMBOL, PERIOD]

# fields from KEY_METRICS
PE_RATIO = "peRatio"
FREE_CASH_FLOW_YIELD = "freeCashFlowYield"
DEBT_TO_EQUITY = "debtToEquity"
INTEREST_COVERAGE = "interestCoverage"
DIVIDEND_YIELD = "dividendYield"
PAYOUT_RATIO = "payoutRatio"

# fields from INCOME_STATEMENT
REVENUE = "revenue"
GROSS_PROFIT = "grossProfit"
GROSS_PROFIT_RATIO = "grossProfitRatio"
OPERATING_INCOME = "operatingIncome"
OPERATING_INCOME_RATIO = "operatingIncomeRatio"
NET_INCOME = "netIncome"
NET_INCOME_RATIO = "netIncomeRatio"
EPS_DILUTED = "epsdiluted"
WEIGHTED_AVERAGE_SHARES_OUTSTANDING_DILUTED = "weightedAverageShsOutDil"
COST_OF_REVENUE = "costOfRevenue"
OPERATING_EXPENSES = "operatingExpenses"

# fields from BALANCE_SHEET_STATEMENT
CASH_AND_CASH_EQUIVALENTS = "cashAndCashEquivalents"
TOTAL_ASSETS = "totalAssets"
TOTAL_CURRENT_LIABILITIES = "totalCurrentLiabilities"

# fields from CASH_FLOW_STATEMENT
FREE_CASH_FLOW = "freeCashFlow"
STOCK_BASED_COMPENSATION = "stockBasedCompensation"

# fields from ENTERPRISE_VALUES
MARKET_CAPITALIZATION = "marketCapitalization"
NUMBER_OF_SHARES = "numberOfShares"

# calculated fields
GROSS_MARGIN = "grossMargin"
OPERATING_MARGIN = "operatingMargin"
NET_MARGIN = "netMargin"
ROCE = "roce"
FREE_CASH_FLOW_ADJUSTED = "freeCashFlowAdjusted"
FREE_CASH_FLOW_YIELD_ADJUSTED = "freeCashFlowYieldAdjusted"
FREE_CASH_FLOW_ADJUSTED_PER_SHARE = "freeCashFlowAdjustedPerShare"
FREE_CASH_FLOW_PER_SHARE = "freeCashFlowPerShare"
STOCK_BASED_COMPENSATION_AS_PCT_OF_FCF = "stockBasedCompensationAsPctOfFcf"

# other constants
FY = "FY"
QUARTER = "quarter"

# constants
SUPPORTED_BASE_TTM_METRICS_LIST = [REVENUE, GROSS_PROFIT, OPERATING_INCOME, NET_INCOME, COST_OF_REVENUE,
                                   OPERATING_EXPENSES, FREE_CASH_FLOW, STOCK_BASED_COMPENSATION, NUMBER_OF_SHARES]
SUPPORTED_TTM_METRICS_LIST = SUPPORTED_BASE_TTM_METRICS_LIST + [GROSS_MARGIN, OPERATING_MARGIN, NET_MARGIN, ROCE,
                                                                FREE_CASH_FLOW_ADJUSTED, FREE_CASH_FLOW_YIELD_ADJUSTED,
                                                                TOTAL_ASSETS, TOTAL_CURRENT_LIABILITIES,
                                                                FREE_CASH_FLOW_PER_SHARE,
                                                                FREE_CASH_FLOW_ADJUSTED_PER_SHARE]
METRIC_FORMAT_DICT = {EPS_DILUTED: "2f", DIVIDEND_YIELD: "2f"}
