import numpy as np

def epirank(od_matrix, damping=0.95, daytime=0.5, max_iter=100, tol=1e-6):
    """
    Implements the EpiRank algorithm for computing epidemic risk in commuting networks.
    
    Parameters:
        od_matrix (numpy.ndarray): Origin-destination (OD) matrix (n x n).
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
    
    # Initialize scores
    scores = np.full(n, 1.0 / n)
    external_factors = np.full(n, 1.0 / n)
    
    for iteration in range(max_iter):
        forward_effect = daytime * (W @ scores)
        backward_effect = (1 - daytime) * (W_T @ scores)
        new_scores = damping * (forward_effect + backward_effect) + (1 - damping) * external_factors
        
        # Check for convergence
        if np.linalg.norm(new_scores - scores, ord=1) < tol:
            break
        scores = new_scores
    
    return scores

# Example usage
if __name__ == "__main__":
    # Example OD matrix (replace with your actual data)
    od_matrix = np.array([
        [0, 50, 20, 0],
        [50, 0, 30, 10],
        [20, 30, 0, 40],
        [0, 10, 40, 0]
    ])
    
    scores = epirank(od_matrix, damping=0.95, daytime=0.5)
    print("EpiRank scores:", scores)
