import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class NMFWithConvergence:
    """
    NMF implementation with convergence tracking using sklearn
    """
    
    def __init__(self, n_components, init='nndsvd', max_iter=200, tol=1e-4, random_state=42):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.convergence_history = []
        
    def fit_transform_with_convergence(self, V):
        """
        Fit NMF model and track convergence
        
        Parameters:
        V: input matrix
        
        Returns:
        W: basis matrix
        H: coefficient matrix
        convergence_history: list of reconstruction errors
        """
        
        # Initialize NMF model with different max_iter values to track convergence
        self.convergence_history = []
        
        # Track convergence by running NMF with increasing iterations
        for iter_count in range(1, self.max_iter + 1, 5):  # Every 5 iterations
            model = NMF(
                n_components=self.n_components,
                init=self.init,
                max_iter=iter_count,
                tol=self.tol,
                random_state=self.random_state,
                verbose=0
            )
            
            W = model.fit_transform(V)
            H = model.components_
            
            # Calculate reconstruction error
            V_approx = np.dot(W, H)
            error = np.linalg.norm(V - V_approx, 'fro') ** 2 / V.size
            self.convergence_history.append(error)
            
            # Early stopping if converged
            if len(self.convergence_history) > 1:
                improvement = abs(self.convergence_history[-2] - self.convergence_history[-1])
                if improvement < self.tol:
                    break
        
        # Final model with full iterations
        final_model = NMF(
            n_components=self.n_components,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=0
        )
        
        W_final = final_model.fit_transform(V)
        H_final = final_model.components_
        
        return W_final, H_final, self.convergence_history

def calculate_nmf_convergence_improved(V, rank, max_iter=200, tol=1e-6, init='nndsvd'):
    """
    Improved: Calculate NMF convergence using sklearn
    
    Parameters:
    V: Input matrix
    rank: Number of components
    max_iter: Maximum number of iterations
    tol: Stopping threshold
    init: Initialization method
    
    Returns:
    errors: List of reconstruction errors
    W: Final basis matrix
    H: Final coefficient matrix
    """
    
    # Use improved class
    nmf_tracker = NMFWithConvergence(
        n_components=rank,
        init=init,
        max_iter=max_iter,
        tol=tol,
        random_state=42
    )
    
    W, H, errors = nmf_tracker.fit_transform_with_convergence(V)
    
    # Plot convergence chart
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    iterations = range(1, len(errors) * 5, 5)[:len(errors)]
    plt.plot(iterations, errors, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error')
    plt.title('NMF Convergence (sklearn implementation)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization of changes
    
    # Improvement plot
    plt.subplot(2, 1, 2)
    if len(errors) > 1:
        improvements = [abs(errors[i-1] - errors[i]) for i in range(1, len(errors))]
        plt.plot(iterations[1:], improvements, 'r-', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Improvement Rate')
        plt.title('Improvement Rate per Iteration')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Display summary information
    print(f"\n{'='*50}")
    print(f"NMF Results Summary")
    print(f"{'='*50}")
    print(f"Input matrix shape: {V.shape}")
    print(f"Decomposition rank: {rank}")
    print(f"Initialization method: {init}")
    print(f"Number of iterations performed: {len(errors)}")
    print(f"Initial error: {errors[0]:.6f}")
    print(f"Final error: {errors[-1]:.6f}")
    if len(errors) > 1:
        improvement_percent = ((errors[0] - errors[-1]) / errors[0]) * 100
        print(f"Improvement percentage: {improvement_percent:.2f}%")
    print(f"W matrix shape: {W.shape}")
    print(f"H matrix shape: {H.shape}")
    
    return errors, W, H

# Example usage:
if __name__ == "__main__":
    print("Testing improved NMF implementation with sklearn")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    V = np.abs(np.random.randn(30, 20))  # Positive matrix
    rank = 5
    
    print(f"Input matrix shape: {V.shape}")
    print(f"Decomposition rank: {rank}")
    print("\nStarting convergence calculation...\n")
    
    # Calculate NMF convergence with different methods
    
    # 1. NNDSVD method (recommended)
    print("1. Using NNDSVD initialization:")
    errors_nndsvd, W_nndsvd, H_nndsvd = calculate_nmf_convergence_improved(
        V, rank, max_iter=100, tol=1e-6, init='nndsvd'
    )
    
    # 2. Random method for comparison
    print("\n2. Using Random initialization:")
    errors_random, W_random, H_random = calculate_nmf_convergence_improved(
        V, rank, max_iter=100, tol=1e-6, init='random'
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison of initialization methods:")
    print("=" * 60)
    print(f"NNDSVD - Final error: {errors_nndsvd[-1]:.8f}")
    print(f"Random - Final error: {errors_random[-1]:.8f}")
    
    if errors_nndsvd[-1] < errors_random[-1]:
        print("✅ NNDSVD performed better")
    else:
        print("✅ Random performed better")
    
    # Check reconstruction quality
    V_reconstructed_nndsvd = np.dot(W_nndsvd, H_nndsvd)
    reconstruction_error = np.linalg.norm(V - V_reconstructed_nndsvd, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(V, 'fro')
    
    print(f"\nRelative reconstruction error: {relative_error:.4f} ({relative_error*100:.2f}%)")
    
    if relative_error < 0.1:
        print("✅ Excellent reconstruction quality")
    elif relative_error < 0.3:
        print("✅ Good reconstruction quality")
    else:
        print("⚠️ Reconstruction quality can be improved")
