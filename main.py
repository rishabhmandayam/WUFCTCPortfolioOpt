import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_files = ["CSVs/DAG_part1.csv", "CSVs/KOP_part1.csv", "CSVs/MON_part1.csv", "CSVs/PED_part1.csv", "CSVs/PUG_part1.csv", "CSVs/TAW_part1.csv", "CSVs/TOW_part1.csv", "CSVs/YON_part1.csv"]
    if len(csv_files) < 8:
        raise ValueError("At least 8 CSV files are required.")

    price_series = []
    for file in csv_files[:8]:
        df = pd.read_csv(file)
        price_series.append(df['C'].rename(file))

    prices = pd.concat(price_series, axis=1).sort_index()

    returns = prices.pct_change().dropna()

    portfolio_returns = []
    for date, row in returns.iterrows():
        w = weights(row)
        port_return = np.dot(w, row.values)
        portfolio_returns.append(port_return)

    portfolio_returns = pd.Series(portfolio_returns, index=returns.index)

    pnl = (1 + portfolio_returns).cumprod()

    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

    print("Final PnL:", pnl.iloc[-1])
    print("Sharpe Ratio:", sharpe_ratio)

    plt.plot(pnl)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title("Portfolio PnL")
    plt.show()


def weights(row):
    """
       Come up with some strategy to determine the weights of each asset in the portfolio.
       Keep in mind that some assets might be heavily correlated,
       some might be inversely correlated, and some might be entirely independent.

       Grading will be done based on Sharpe ratio (a measurement of risk-adjusted returns).
       Also, keep in mind that you will NOT be graded on this data set --
       it will be another 1000 candles drawn from the same distribution. This means
       that while yes, you can look at future results in this dataset to make infinite money,
       it will likely not be beneficial to do so.

       Good luck everyone!
       """
    DAG_df = pd.read_csv("CSVs/DAG_part1.csv")
    KOP_df = pd.read_csv("CSVs/KOP_part1.csv")
    MON_df = pd.read_csv("CSVs/MON_part1.csv")
    PED_df = pd.read_csv("CSVs/PED_part1.csv")
    PUG_df = pd.read_csv("CSVs/PUG_part1.csv")
    TAW_df = pd.read_csv("CSVs/TAW_part1.csv")
    TOW_df = pd.read_csv("CSVs/TOW_part1.csv")
    YON_df = pd.read_csv("CSVs/YON_part1.csv")
    # Above are pandas data frames for the individual stocks. We recommend using these to calculate the weights.

    # This is just a simple example where all assets are equally weighted.
    number_assets = 8
    weights = [1/number_assets for _ in range(number_assets)]

    # please return the weights as a numpy array
    return np.array(weights)



main()
