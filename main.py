import numpy as np

def epirank_with_factors(od_matrix, climate, density, economy, weights, damping=0.95, daytime=0.5, max_iter=100, tol=1e-6):
    """
    EpiRank algorithm with external factors: climate, population density, and economy.
    
    Parameters:
        od_matrix (numpy.ndarray): Origin-destination (OD) matrix (n x n).
        climate (numpy.ndarray): Climate factor array (size n).
        density (numpy.ndarray): Population density array (size n).
        economy (numpy.ndarray): Economic condition array (size n).
        weights (tuple): Weights for climate, density, and economy (w1, w2, w3).
        damping (float): Damping factor for topological structure (default 0.95).
        daytime (float): Forward movement weighting (0 to 1).
        max_iter (int): Maximum number of iterations (default 100).
        tol (float): Convergence tolerance (default 1e-6).

    Returns:
        numpy.ndarray: EpiRank scores for each node.
    """
    n = od_matrix.shape[0]
    
    # Normalize OD matrix
    W = od_matrix / od_matrix.sum(axis=1, keepdims=True)
    W_T = W.T
    
    # Calculate external factors
    exFac = weights[0] * climate + weights[1] * density + weights[2] * economy
    exFac = exFac / exFac.sum()  # Normalize to sum up to 1

    # Initialize scores
    scores = np.full(n, 1.0 / n)

    for iteration in range(max_iter):
        forward_effect = daytime * (W @ scores)
        backward_effect = (1 - daytime) * (W_T @ scores)
        new_scores = damping * (forward_effect + backward_effect) + (1 - damping) * exFac
        
        # Check for convergence
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            break
        scores = new_scores
    
    return scores

# Example usage
if __name__ == "__main__":
    # Example OD matrix
    od_matrix = np.array([
        [0, 50, 20, 0],
        [50, 0, 30, 10],
        [20, 30, 0, 40],
        [0, 10, 40, 0]
    ])
    
    # Example external factors
    climate = np.array([0.8, 0.6, 0.9, 0.5])  # Normalized
    density = np.array([0.7, 0.8, 0.6, 0.9])  # Normalized
    economy = np.array([0.6, 0.7, 0.5, 0.8])  # Normalized
    weights = (0.4, 0.4, 0.2)  # Weights for factors

    scores = epirank_with_factors(od_matrix, climate, density, economy, weights)
    print("EpiRank scores with factors:", scores)
