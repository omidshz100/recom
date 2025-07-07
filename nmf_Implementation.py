import numpy as np

def random_initialization(A, rank):
    """
    Initialize matrices W and H randomly.

    Parameters:
    - A: Input matrix
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix
    - H: Initialized H matrix
    """
    num_docs = A.shape[0]
    num_terms = A.shape[1]
    W = np.random.uniform(1, 2, (num_docs, rank))
    H = np.random.uniform(1, 2, (rank, num_terms))
    return W, H

"""
## What NNDSVD Does:
Instead of random initialization, NNDSVD uses the mathematical 
structure of the original matrix A to create better starting points 
for W and H, leading to faster convergence and better results.

- Decomposes matrix A into: A = U × S × V^T
- u : Left singular vectors (columns are eigenvectors)
- s : Singular values (diagonal of S matrix)
- v : Right singular vectors (transposed)


"""
def nndsvd_initialization(A, rank):
    """
    Initialize matrices W and H using Non-negative 
        Double Singular Value Decomposition (NNDSVD).

    Parameters:
    - A: Input matrix
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix
    - H: Initialized H matrix
    """
    u, s, v = np.linalg.svd(A, full_matrices=False)
    v = v.T
    # Creates zero matrices with correct dimensions
    w = np.zeros((A.shape[0], rank))
    h = np.zeros((rank, A.shape[1]))

    """
    - Uses the largest singular value s[0] and corresponding vectors
    - Takes absolute values to ensure non-negativity
    - Scales by square root of singular value
    """
    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

    for i in range(1, rank):
        ui = u[:, i] # i-th left singular vector
        vi = v[:, i] # i-th right singular vector
        # Split into Positive and Negative Parts
        ui_pos = (ui >= 0) * ui # Keep only positive elements
        ui_neg = (ui < 0) * -ui # Keep only negative elements (made positive)
        vi_pos = (vi >= 0) * vi # Keep only positive elements
        vi_neg = (vi < 0) * -vi # Keep only negative elements (made positive)

        # Calculate Norms
        ui_pos_norm = np.linalg.norm(ui_pos, 2) # L2 norm of positive part
        ui_neg_norm = np.linalg.norm(ui_neg, 2)  # L2 norm of negative part
        vi_pos_norm = np.linalg.norm(vi_pos, 2) # L2 norm of positive part
        vi_neg_norm = np.linalg.norm(vi_neg, 2) # L2 norm of negative part

        # Choose Best Combination
        norm_pos = ui_pos_norm * vi_pos_norm # Strength of positive combination
        norm_neg = ui_neg_norm * vi_neg_norm # Strength of negative combination

        if norm_pos >= norm_neg:
             # Use positive parts
            w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
            h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
        else:
            # Use negative parts (made positive)
            w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
            h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T

    return w, h

def multiplicative_update(A, k, max_iter, init_mode='random'):
    """
    Perform Multiplicative Update (MU) algorithm for Non-negative 
            Matrix Factorization (NMF).

    Parameters:
    - A: Input matrix
    - k: Rank of the factorization
    - max_iter: Maximum number of iterations
    - init_mode: Initialization mode ('random' or 'nndsvd')

    Returns:
    - W: Factorized matrix W
    - H: Factorized matrix H
    - norms: List of Frobenius norms at each iteration
    """
    if init_mode == 'random':
        W, H = random_initialization(A, k)
    elif init_mode == 'nndsvd':
        W, H = nndsvd_initialization(A, k)

    norms = []
    # epsilon (1e-10) : Prevents division by zero
    epsilon = 1.0e-10
    # The @ symbol in Python is the 
    # matrix multiplication operator , introduced in Python 3.5 (PEP 465).
    for _ in range(max_iter):
        # Update H
        W_TA = W.T @ A
        W_TWH = W.T @ W @ H + epsilon
        H *= W_TA / W_TWH

        # Update W
        AH_T = A @ H.T
        WHH_T = W @ H @ H.T + epsilon
        W *= AH_T / WHH_T

        norm = np.linalg.norm(A - W @ H, 'fro')
        norms.append(norm)

    return W, H, norms



# Improved examples with better formatting and readability
def print_matrix_info(matrix, name, decimals=3, sizeCondition=25):
    """Print matrix information in a readable format"""
    print(f"\n{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Min: {matrix.min():.{decimals}f}, Max: {matrix.max():.{decimals}f}")
    print(f"  Mean: {matrix.mean():.{decimals}f}, Std: {matrix.std():.{decimals}f}")
    if matrix.size <= sizeCondition:  # Only show full matrix if small
        print(f"  Values:\n{np.round(matrix, decimals)}")
    else:
        print(f"  First 3x3 corner:\n{np.round(matrix[:3, :3], decimals)}")

def plot_convergence(norms, title="NMF Convergence"):
    """Plot convergence if matplotlib is available"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(norms)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Frobenius Norm')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        print(f"Convergence values (first 10): {norms[:10]}")
        print(f"Final convergence: {norms[-1]:.6f}")

if __name__ == "__main__":
    print("=" * 60)
    print("NON-NEGATIVE MATRIX FACTORIZATION (NMF) DEMO")
    print("=" * 60)
    
    # Create test matrix
    np.random.seed(42)  # For reproducible results
    A = np.random.rand(6, 8)  # Smaller matrix for better readability
    print_matrix_info(A, "Original Matrix A", sizeCondition=800)
    
    rank = 3
    print(f"\nFactorization rank: {rank}")
    
    # Test 1: Random Initialization
    print("\n" + "-" * 40)
    print("1. RANDOM INITIALIZATION")
    print("-" * 40)
    W_rand, H_rand = random_initialization(A, rank)
    print_matrix_info(W_rand, "W (Random)")
    print_matrix_info(H_rand, "H (Random)")
    
    # Reconstruction error
    reconstruction_rand = W_rand @ H_rand
    error_rand = np.linalg.norm(A - reconstruction_rand, 'fro')
    print(f"\nReconstruction error: {error_rand:.6f}")
    
    # Test 2: NNDSVD Initialization
    print("\n" + "-" * 40)
    print("2. NNDSVD INITIALIZATION")
    print("-" * 40)
    W_nndsvd, H_nndsvd = nndsvd_initialization(A, rank)
    print_matrix_info(W_nndsvd, "W (NNDSVD)")
    print_matrix_info(H_nndsvd, "H (NNDSVD)")
    
    # Reconstruction error
    reconstruction_nndsvd = W_nndsvd @ H_nndsvd
    error_nndsvd = np.linalg.norm(A - reconstruction_nndsvd, 'fro')
    print(f"\nReconstruction error: {error_nndsvd:.6f}")
    print_matrix_info(reconstruction_nndsvd, "NNDSVD Reconstruction Matrix", sizeCondition=800)

    # Test 3: Multiplicative Update
    print("\n" + "-" * 40)
    print("3. MULTIPLICATIVE UPDATE ALGORITHM")
    print("-" * 40)
    max_iter = 50  # Reduced for faster execution
    W_mu, H_mu, norms = multiplicative_update(A, rank, max_iter, init_mode='random')
    
    print_matrix_info(W_mu, "W (After MU)")
    print_matrix_info(H_mu, "H (After MU)")
    
    # Final reconstruction
    reconstruction_mu = W_mu @ H_mu
    final_error = norms[-1]
    print(f"\nFinal reconstruction error: {final_error:.6f}")
    print(f"Improvement: {norms[0]:.6f} → {final_error:.6f} ({((norms[0]-final_error)/norms[0]*100):.1f}% reduction)")
    
    # Show convergence
    print(f"\nConvergence (every 10th iteration):")
    for i in range(0, len(norms), 10):
        print(f"  Iteration {i:2d}: {norms[i]:.6f}")
    
    # Plot convergence if possible
    print("\n" + "-" * 40)
    print("4. CONVERGENCE VISUALIZATION")
    print("-" * 40)
    plot_convergence(norms)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

