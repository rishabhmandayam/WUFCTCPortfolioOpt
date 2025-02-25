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
    
    # Convert to returns
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

    def markowitz_optimization(returns_matrix, avg_returns_array, start_idx=-100, end_idx=None):
    # Use data from start_idx to end_idx
    # Negative start_idx counts from the end
    # end_idx=None means use all data up to the end
        window_returns = returns_matrix[start_idx:end_idx]
        window_cov = np.cov(window_returns.T)
        
        # Calculate inverse of covariance matrix
        inv_cov = np.linalg.inv(window_cov)
        
        # Calculate weights using closed form solution
        ones = np.ones(len(avg_returns_array))
        numerator = np.dot(inv_cov, ones)
        denominator = np.dot(ones, numerator)
        
        # Global minimum variance portfolio weights
        weights = numerator / denominator
        
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)  # Floor at 0
        weights = weights / np.sum(weights)  # Normalize to sum to 1
        
        return weights

    DAG_df = pd.read_csv("CSVs/DAG_part1.csv")
    KOP_df = pd.read_csv("CSVs/KOP_part1.csv")
    MON_df = pd.read_csv("CSVs/MON_part1.csv")
    PED_df = pd.read_csv("CSVs/PED_part1.csv")
    PUG_df = pd.read_csv("CSVs/PUG_part1.csv")
    TAW_df = pd.read_csv("CSVs/TAW_part1.csv")
    TOW_df = pd.read_csv("CSVs/TOW_part1.csv")
    YON_df = pd.read_csv("CSVs/YON_part1.csv")

    DAG_returns = (DAG_df['O'] - DAG_df['O'].shift(1)) / DAG_df['O'].shift(1) * 100
    KOP_returns = (KOP_df['O'] - KOP_df['O'].shift(1)) / KOP_df['O'].shift(1) * 100
    MON_returns = (MON_df['O'] - MON_df['O'].shift(1)) / MON_df['O'].shift(1) * 100
    PED_returns = (PED_df['O'] - PED_df['O'].shift(1)) / PED_df['O'].shift(1) * 100
    PUG_returns = (PUG_df['O'] - PUG_df['O'].shift(1)) / PUG_df['O'].shift(1) * 100
    TAW_returns = (TAW_df['O'] - TAW_df['O'].shift(1)) / TAW_df['O'].shift(1) * 100
    TOW_returns = (TOW_df['O'] - TOW_df['O'].shift(1)) / TOW_df['O'].shift(1) * 100
    YON_returns = (YON_df['O'] - YON_df['O'].shift(1)) / YON_df['O'].shift(1) * 100

    # Calculate average daily returns
    avg_returns = {
        'DAG': DAG_returns.mean(),
        'KOP': KOP_returns.mean(),
        'MON': MON_returns.mean(),
        'PED': PED_returns.mean(),
        'PUG': PUG_returns.mean(),
        'TAW': TAW_returns.mean(),
        'TOW': TOW_returns.mean(),
        'YON': YON_returns.mean()
    }
    # Convert average returns to numpy array
    avg_returns_array = np.array(list(avg_returns.values()))

    # Create returns matrix for covariance calculation
    returns_matrix = np.column_stack([
        DAG_returns,
        KOP_returns, 
        MON_returns,
        PED_returns,
        PUG_returns,
        TAW_returns,
        TOW_returns,
        YON_returns
    ])


    optimal_weights = markowitz_optimization(returns_matrix, avg_returns_array, -30, -5)
    #print("\nOptimal Weights:")
    
    #print(optimal_weights)


    return np.array([0.0, 0.27752484, 0.16219052, 0.0, 0.2002473, 0.02447719, 0.14059584, 0.19496432])
    #return np.array(optimal_weights)



main()
