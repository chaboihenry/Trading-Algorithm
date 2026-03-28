# adapated from github.com/chaboihenry/Portfolio-Optimizer

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

# ------ Core Math Extracted From Portfolio Optimizer Repo --------

def calculate_covariance_matrix(returns: pd.DataFrame):
    # use raw covariance since HRP is scale-invariant
    return returns.cov()

def calculate_correlation_matrix(returns: pd.DataFrame):
    return returns.corr()

def calculate_distance_matrix(correlation_matrix: pd.DataFrame):
    # converts correlation into physical distance for tree graph
    return np.sqrt(0.5 * (1 - correlation_matrix))

def generate_cluster_linkage(distance_matrix: pd.DataFrame):
    return linkage(squareform(distance_matrix), method='single')

def get_quasi_diag(linkage_matrix: np.ndarray, num_assets: int):
    # reorders cov matrix so highly corr spreads sit next to each other
    sort_indices = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])

    while sort_indices.max() >= num_assets:
        sort_indices.index = range(0, sort_indices.shape[0] * 2, 2)
        df0 = sort_indices[sort_indices >= num_assets]
        i = df0.index
        j = (df0.values - num_assets).astype(int)
        sort_indices[i] = linkage_matrix[j, 0]
        df1 = pd.Series(linkage_matrix[j, 1], index=i + 1)
        sort_indices = pd.concat([sort_indices, df1])
        sort_indices = sort_indices.sort_index()
        sort_indices.index = range(sort_indices.shape[0])

    return sort_indices.astype(int).tolist()

def get_cluster_var(cov_matrix: pd.DataFrame, cluster_items: list[int]):
    # calculates the var of a specific branch on the tree
    cluster_cov = cov_matrix.loc[cluster_items, cluster_items]
    inv_diag = 1.0 / np.diag(cluster_cov)
    cluster_weights = inv_diag / np.sum(inv_diag)
    cluster_variance = np.dot(cluster_weights.T, np.dot(cluster_cov, cluster_weights))

    return cluster_variance

def get_rec_bipartition(cov_matrix: pd.DataFrame, sorted_indices: list[str]):
    # recursive top-down split, allocating capital inversely to variance
    weights = pd.Series(1.0, index=sorted_indices)
    clusters = [sorted_indices]

    while len(clusters) > 0:
        current_cluster = clusters.pop(0)

        if len(current_cluster) > 1:
            midpoint = len(current_cluster) // 2
            left_cluster = current_cluster[:midpoint]
            right_cluster = current_cluster[midpoint:]
            
            left_var = get_cluster_var(cov_matrix, left_cluster)
            right_var = get_cluster_var(cov_matrix, right_cluster)
            
            alpha = 1.0 - (left_var / (left_var + right_var))
            
            weights[left_cluster] *= alpha
            weights[right_cluster] *= (1.0 - alpha)
            
            clusters.append(left_cluster)
            clusters.append(right_cluster)
    
    return weights 

# ------ Allocation API for the Master Pipeline --------

def allocate_capital(spread_returns: pd.DataFrame, total_risk_budget: float):
    # input = historical returns of approved spreads and total kelly budget
    # output = dict of exact dollar allocations per spread
    print(f"\n ALLOCATOR: Initiating HRP Clustering for {spread_returns.shape[1]} approved spreads...")

    num_assets = spread_returns.shape[1]

    # edge case: if only 1 spread is approved, it gets the entire budget
    if num_assets == 1:
        spread_name = spread_returns.columns[0]
        return {spread_name: total_risk_budget}
    
    # 1. Calculate dependencies
    cov_matrix = calculate_covariance_matrix(spread_returns)
    corr_matrix = calculate_correlation_matrix(spread_returns)
    dist_matrix = calculate_distance_matrix(corr_matrix)

    # 2. build the tree
    linkage_matrix = generate_cluster_linkage(dist_matrix)

    # 3. sort and allocate
    sorted_indices = get_quasi_diag(linkage_matrix, num_assets)
    sorted_spreads = spread_returns.columns[sorted_indices].tolist()

    hrp_weights = get_rec_bipartition(cov_matrix, sorted_spreads)

    # 4. translate abstract weights (sum to 1.0) into hard dollar amounts
    dollar_allocations = (hrp_weights * total_risk_budget).to_dict()

    return dollar_allocations


if __name__ == "__main__":
    # mocking the data structure passed from the_filter
    print("===== Testing HRP Spread Allocator ====")

    # simulated 5-min bar returns for 4 different approved cointegrated spreads
    np.random.seed(42)
    mock_returns = pd.DataFrame({
      'V_MA_Spread': np.random.normal(0.0001, 0.002, 1000),
        'AAPL_MSFT_Spread': np.random.normal(0.0001, 0.003, 1000),
        'NVDA_AMD_Spread': np.random.normal(0.0002, 0.006, 1000), # Very volatile
        'JPM_BAC_Spread': np.random.normal(0.00005, 0.001, 1000)   # Very stable
    })

    # assume the Meta-Labeler generated a combined Half-Kelly budget of $25.00
    combined_kelly_budget = 25.00

    final_allocations = allocate_capital(mock_returns, combined_kelly_budget)

    print(f"\nTotal Capital to Deploy: ${combined_kelly_budget:.2f}")
    print("=== Final Dollar Allocators Per Spread====")
    for spread, amount in final_allocations.items():
        weight_pct = (amount / combined_kelly_budget) * 100
        print(f"{spread}: ${amount:.2f} ({weight_pct:.1f}% of budget)")



