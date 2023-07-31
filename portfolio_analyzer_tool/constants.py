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
INDEX_KEYS_LIST = [DATE, SYMBOL, PERIOD]

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

# calculated fields
GROSS_MARGIN = "grossMargin"
OPERATING_MARGIN = "operatingMargin"
NET_MARGIN = "netMargin"
ROCE = "roce"
FREE_CASH_FLOW_ADJUSTED = "freeCashFlowAdjusted"
FREE_CASH_FLOW_YIELD_ADJUSTED = "freeCashFlowYieldAdjusted"

# other constants
FY = "FY"
