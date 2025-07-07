import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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

def nndsvd_initialization(A, rank):
    """
    Initialize matrices W and H using Non-negative Double Singular Value Decomposition (NNDSVD).

    Parameters:
    - A: Input matrix
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix
    - H: Initialized H matrix
    """
    # Ensure rank doesn't exceed matrix dimensions
    max_rank = min(A.shape)
    if rank >= max_rank:
        rank = max_rank - 1
        print(f"Warning: Rank reduced to {rank} due to matrix dimensions")
    
    u, s, v = np.linalg.svd(A, full_matrices=False)
    v = v.T
    w = np.zeros((A.shape[0], rank))
    h = np.zeros((rank, A.shape[1]))

    # Handle the case where we have fewer singular values than requested rank
    available_components = min(rank, len(s))
    
    if available_components > 0:
        w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
        h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

        for i in range(1, available_components):
            ui = u[:, i]
            vi = v[:, i]
            ui_pos = (ui >= 0) * ui
            ui_neg = (ui < 0) * -ui
            vi_pos = (vi >= 0) * vi
            vi_neg = (vi < 0) * -vi

            ui_pos_norm = np.linalg.norm(ui_pos, 2)
            ui_neg_norm = np.linalg.norm(ui_neg, 2)
            vi_pos_norm = np.linalg.norm(vi_pos, 2)
            vi_neg_norm = np.linalg.norm(vi_neg, 2)

            norm_pos = ui_pos_norm * vi_pos_norm
            norm_neg = ui_neg_norm * vi_neg_norm

            if norm_pos >= norm_neg and ui_pos_norm > 0 and vi_pos_norm > 0:
                w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
                h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
            elif ui_neg_norm > 0 and vi_neg_norm > 0:
                w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
                h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T
            else:
                # Fallback to small random values
                w[:, i] = np.random.uniform(0.01, 0.1, A.shape[0])
                h[i, :] = np.random.uniform(0.01, 0.1, A.shape[1])
    
    # Fill remaining components with small random values if needed
    for i in range(available_components, rank):
        w[:, i] = np.random.uniform(0.01, 0.1, A.shape[0])
        h[i, :] = np.random.uniform(0.01, 0.1, A.shape[1])

    return w, h

def multiplicative_update(A, k, max_iter, init_mode='random'):
    """
    Perform Multiplicative Update (MU) algorithm for Non-negative Matrix Factorization (NMF).

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
    epsilon = 1.0e-10
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
def print_matrix_info(matrix, name, decimals=3):
    """Print matrix information in a readable format"""
    print(f"\n{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Min: {matrix.min():.{decimals}f}, Max: {matrix.max():.{decimals}f}")
    print(f"  Mean: {matrix.mean():.{decimals}f}, Std: {matrix.std():.{decimals}f}")
    if matrix.size <= 25:  # Only show full matrix if small
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

def load_movielens_data(movies_path, ratings_path, max_users=100, max_movies=200):
    """
    Load and process MovieLens dataset into a user-movie rating matrix.
    
    Parameters:
    - movies_path: Path to movies.csv file
    - ratings_path: Path to ratings.csv file
    - max_users: Maximum number of users to include (for computational efficiency)
    - max_movies: Maximum number of movies to include
    
    Returns:
    - rating_matrix: User-movie rating matrix (numpy array)
    - user_ids: List of user IDs
    - movie_ids: List of movie IDs
    - movie_titles: List of movie titles
    """
    print(f"Loading MovieLens data...")
    
    # Load movies data
    movies_df = pd.read_csv(movies_path)
    print(f"Loaded {len(movies_df)} movies")
    
    # Load ratings data
    ratings_df = pd.read_csv(ratings_path)
    print(f"Loaded {len(ratings_df)} ratings")
    
    # Get top users and movies for computational efficiency
    top_users = ratings_df['userId'].value_counts().head(max_users).index
    top_movies = ratings_df['movieId'].value_counts().head(max_movies).index
    
    # Filter data
    filtered_ratings = ratings_df[
        (ratings_df['userId'].isin(top_users)) & 
        (ratings_df['movieId'].isin(top_movies))
    ]
    
    print(f"Filtered to {len(filtered_ratings)} ratings from {len(top_users)} users and {len(top_movies)} movies")
    
    # Create user and movie mappings
    unique_users = sorted(filtered_ratings['userId'].unique())
    unique_movies = sorted(filtered_ratings['movieId'].unique())
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    
    # Create rating matrix
    num_users = len(unique_users)
    num_movies = len(unique_movies)
    rating_matrix = np.zeros((num_users, num_movies))
    
    # Fill rating matrix
    for _, row in filtered_ratings.iterrows():
        user_idx = user_to_idx[row['userId']]
        movie_idx = movie_to_idx[row['movieId']]
        rating_matrix[user_idx, movie_idx] = row['rating']
    
    # Get movie titles for the selected movies
    movie_titles = []
    for movie_id in unique_movies:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        movie_titles.append(title)
    
    print(f"Created rating matrix of shape {rating_matrix.shape}")
    print(f"Matrix sparsity: {(rating_matrix == 0).sum() / rating_matrix.size * 100:.1f}% zeros")
    
    return rating_matrix, unique_users, unique_movies, movie_titles

if __name__ == "__main__":
    print("=" * 60)
    print("MOVIELENS NMF RECOMMENDATION SYSTEM DEMO")
    print("=" * 60)
    
    # Load MovieLens dataset
    movies_path = "../ml-latest/movies.csv"
    ratings_path = "../ml-latest/ratings.csv"
    
    try:
        # Load and process data
        A, user_ids, movie_ids, movie_titles = load_movielens_data(
            movies_path, ratings_path, max_users=50, max_movies=100
        )
        
        print_matrix_info(A, "MovieLens Rating Matrix")
        
        # Set factorization rank (number of latent factors)
        # Rank must be smaller than the minimum dimension of the matrix
        max_rank = min(A.shape) - 1
        rank = min(10, max_rank)
        print(f"\nFactorization rank (latent factors): {rank}")
        print(f"Matrix shape: {A.shape}, Max possible rank: {max_rank}")
        
        # Test 1: Random Initialization
        print("\n" + "-" * 50)
        print("1. RANDOM INITIALIZATION")
        print("-" * 50)
        W_rand, H_rand = random_initialization(A, rank)
        print_matrix_info(W_rand, "W (User-Factor Matrix)")
        print_matrix_info(H_rand, "H (Factor-Movie Matrix)")
        
        # Reconstruction error
        reconstruction_rand = W_rand @ H_rand
        error_rand = np.linalg.norm(A - reconstruction_rand, 'fro')
        print(f"\nReconstruction error: {error_rand:.6f}")
        
        # Test 2: NNDSVD Initialization
        print("\n" + "-" * 50)
        print("2. NNDSVD INITIALIZATION")
        print("-" * 50)
        W_nndsvd, H_nndsvd = nndsvd_initialization(A, rank)
        print_matrix_info(W_nndsvd, "W (User-Factor Matrix)")
        print_matrix_info(H_nndsvd, "H (Factor-Movie Matrix)")
        
        # Reconstruction error
        reconstruction_nndsvd = W_nndsvd @ H_nndsvd
        error_nndsvd = np.linalg.norm(A - reconstruction_nndsvd, 'fro')
        print(f"\nReconstruction error: {error_nndsvd:.6f}")
        
        # Test 3: Multiplicative Update Algorithm
        print("\n" + "-" * 50)
        print("3. MULTIPLICATIVE UPDATE ALGORITHM")
        print("-" * 50)
        max_iter = 100
        print(f"Running NMF with {max_iter} iterations...")
        W_mu, H_mu, norms = multiplicative_update(A, rank, max_iter, init_mode='nndsvd')
        
        print_matrix_info(W_mu, "W (Final User-Factor Matrix)")
        print_matrix_info(H_mu, "H (Final Factor-Movie Matrix)")
        
        # Final reconstruction and analysis
        reconstruction_mu = W_mu @ H_mu
        final_error = norms[-1]
        print(f"\nFinal reconstruction error: {final_error:.6f}")
        print(f"Improvement: {norms[0]:.6f} → {final_error:.6f} ({((norms[0]-final_error)/norms[0]*100):.1f}% reduction)")
        
        # Show convergence
        print(f"\nConvergence (every 20th iteration):")
        for i in range(0, len(norms), 20):
            print(f"  Iteration {i:3d}: {norms[i]:.6f}")
        
        # Movie recommendation example
        print("\n" + "-" * 50)
        print("4. MOVIE RECOMMENDATION EXAMPLE")
        print("-" * 50)
        
        # Pick a user who has rated some movies
        user_idx = 0  # First user
        user_id = user_ids[user_idx]
        
        print(f"\nRecommendations for User {user_id}:")
        print(f"Original ratings (non-zero only):")
        
        # Show original ratings
        original_ratings = A[user_idx, :]
        rated_movies = np.where(original_ratings > 0)[0]
        
        for movie_idx in rated_movies[:5]:  # Show first 5 rated movies
            rating = original_ratings[movie_idx]
            title = movie_titles[movie_idx]
            print(f"  {title[:50]:<50} Rating: {rating:.1f}")
        
        # Show predicted ratings
        predicted_ratings = reconstruction_mu[user_idx, :]
        
        # Find movies not rated by user but with high predicted ratings
        unrated_movies = np.where(original_ratings == 0)[0]
        if len(unrated_movies) > 0:
            unrated_predictions = predicted_ratings[unrated_movies]
            top_recommendations = unrated_movies[np.argsort(unrated_predictions)[::-1]]
            
            print(f"\nTop 5 recommended movies (unrated by user):")
            for i, movie_idx in enumerate(top_recommendations[:5]):
                predicted_rating = predicted_ratings[movie_idx]
                title = movie_titles[movie_idx]
                print(f"  {i+1}. {title[:50]:<50} Predicted: {predicted_rating:.2f}")
        
        # Plot convergence
        print("\n" + "-" * 50)
        print("5. CONVERGENCE VISUALIZATION")
        print("-" * 50)
        plot_convergence(norms, "MovieLens NMF Convergence")
        
        print("\n" + "=" * 60)
        print("MOVIELENS NMF DEMO COMPLETED")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please ensure the following files exist:")
        print(f"  - {movies_path}")
        print(f"  - {ratings_path}")
        print(f"\nError details: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your data files and try again.")

