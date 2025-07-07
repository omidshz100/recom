import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nmf_Implementation import (
    random_initialization, 
    nndsvd_initialization, 
    multiplicative_update,
    print_matrix_info
)

# Configure Streamlit page
st.set_page_config(
    page_title="NMF Interactive Demo",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🔢 Non-negative Matrix Factorization (NMF) Interactive Demo</h1>', unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.header("🎛️ Configuration")

# Matrix parameters
st.sidebar.subheader("Matrix Settings")
matrix_mode = st.sidebar.selectbox("Matrix Creation Mode", ["Auto Generate", "Manual Input"], index=0)

if matrix_mode == "Auto Generate":
    rows = st.sidebar.slider("Number of Rows", min_value=3, max_value=20, value=6, step=1)
    cols = st.sidebar.slider("Number of Columns", min_value=3, max_value=20, value=8, step=1)
    sparsity = st.sidebar.slider("Sparsity Level (%)", min_value=0, max_value=80, value=30, step=5)
    random_seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=1000, value=42, step=1)
else:
    rows = st.sidebar.slider("Number of Rows", min_value=3, max_value=10, value=4, step=1)
    cols = st.sidebar.slider("Number of Columns", min_value=3, max_value=10, value=5, step=1)
    st.sidebar.info("💡 Enter values 0-10 in the matrix editor below. They will be normalized to 0-1.")

# NMF parameters
st.sidebar.subheader("NMF Settings")
rank = st.sidebar.slider("Factorization Rank", min_value=1, max_value=min(rows, cols)-1, value=3, step=1)
max_iterations = st.sidebar.slider("Max Iterations", min_value=10, max_value=200, value=50, step=10)
init_method = st.sidebar.selectbox("Initialization Method", ["random", "nndsvd"], index=1)

# Generate matrix button
if matrix_mode == "Auto Generate":
    if st.sidebar.button("🔄 Generate New Matrix", type="primary"):
        st.session_state.matrix_generated = True
        st.session_state.matrix_mode = "auto"
else:
    if st.sidebar.button("🔄 Create Manual Matrix", type="primary"):
        st.session_state.matrix_generated = True
        st.session_state.matrix_mode = "manual"

# Initialize session state
if 'matrix_generated' not in st.session_state:
    st.session_state.matrix_generated = True
    st.session_state.matrix_mode = "auto"

if st.session_state.matrix_generated:
    if matrix_mode == "Auto Generate" or st.session_state.matrix_mode == "auto":
        # Generate matrix automatically
        if matrix_mode == "Auto Generate":
            np.random.seed(random_seed)
            A = np.random.rand(rows, cols)
            A[A < sparsity/100] = 0  # Apply sparsity
        else:
            # Use default values for auto mode when switching back
            np.random.seed(42)
            A = np.random.rand(6, 8)
            A[A < 0.3] = 0
    else:
        # Manual matrix input
        st.markdown('<h2 class="section-header">✏️ Manual Matrix Input</h2>', unsafe_allow_html=True)
        st.write("Enter values between 0-10. They will be automatically normalized to 0-1 for NMF processing.")
        
        # Create initial matrix with some sample values
        initial_data = np.ones((rows, cols))  # Start with 1s instead of zeros
        
        # Create editable dataframe
        df_input = pd.DataFrame(
            initial_data,
            columns=[f'Col_{i+1}' for i in range(cols)],
            index=[f'Row_{i+1}' for i in range(rows)]
        )
        
        # Matrix editor with explicit configuration
        edited_df = st.data_editor(
            df_input,
            use_container_width=True,
            num_rows="fixed",
            hide_index=False,
            key=f"matrix_editor_{rows}_{cols}"
        )
        
        # Convert back to numpy and normalize
        manual_matrix_raw = edited_df.values
        
        # Normalize to 0-1 range
        max_val = np.max(manual_matrix_raw)
        if max_val > 0:
            A = manual_matrix_raw / 10.0  # Normalize assuming max input is 10
        else:
            A = manual_matrix_raw
        
        # Show normalization info
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Values (0-10):**")
            st.write(f"Min: {manual_matrix_raw.min():.1f}, Max: {manual_matrix_raw.max():.1f}")
        with col2:
            st.write("**Normalized Values (0-1):**")
            st.write(f"Min: {A.min():.3f}, Max: {A.max():.3f}")
    
    # Display matrix information
    if matrix_mode == "Auto Generate" or st.session_state.matrix_mode == "auto":
        st.markdown('<h2 class="section-header">📊 Generated Matrix A</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="section-header">📊 Your Custom Matrix A (Normalized)</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shape", f"{A.shape[0]} × {A.shape[1]}")
        st.metric("Sparsity", f"{np.sum(A == 0) / A.size * 100:.1f}%")
    
    with col2:
        st.metric("Min Value", f"{A.min():.3f}")
        st.metric("Max Value", f"{A.max():.3f}")
    
    with col3:
        st.metric("Mean", f"{A.mean():.3f}")
        st.metric("Std Dev", f"{A.std():.3f}")
    
    # Display matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(A, cmap='viridis', aspect='auto')
    ax.set_title('Original Matrix A', fontsize=14, fontweight='bold')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
    
    # Show matrix values if small enough
    if A.size <= 100:
        st.subheader("Matrix Values")
        df = pd.DataFrame(A, columns=[f'Col_{i+1}' for i in range(A.shape[1])], 
                         index=[f'Row_{i+1}' for i in range(A.shape[0])])
        st.dataframe(df.round(3), use_container_width=True)
    
    # Run NMF Analysis
    st.markdown('<h2 class="section-header">🔬 NMF Analysis</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Run NMF Analysis", type="primary"):
        with st.spinner('Running NMF analysis...'):
            # Test different initialization methods
            results = {}
            
            # Random initialization
            W_rand, H_rand = random_initialization(A, rank)
            reconstruction_rand = W_rand @ H_rand
            error_rand = np.linalg.norm(A - reconstruction_rand, 'fro')
            results['Random'] = {
                'W': W_rand, 'H': H_rand, 
                'reconstruction': reconstruction_rand,
                'error': error_rand
            }
            
            # NNDSVD initialization
            W_nndsvd, H_nndsvd = nndsvd_initialization(A, rank)
            reconstruction_nndsvd = W_nndsvd @ H_nndsvd
            error_nndsvd = np.linalg.norm(A - reconstruction_nndsvd, 'fro')
            results['NNDSVD'] = {
                'W': W_nndsvd, 'H': H_nndsvd,
                'reconstruction': reconstruction_nndsvd,
                'error': error_nndsvd
            }
            
            # Multiplicative Update
            W_mu, H_mu, norms = multiplicative_update(A, rank, max_iterations, init_mode=init_method)
            reconstruction_mu = W_mu @ H_mu
            results['Multiplicative Update'] = {
                'W': W_mu, 'H': H_mu,
                'reconstruction': reconstruction_mu,
                'error': norms[-1],
                'convergence': norms
            }
            
            # Display results
            st.markdown('<h3 class="section-header">📈 Reconstruction Errors</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Random Init", 
                    f"{results['Random']['error']:.6f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "NNDSVD Init", 
                    f"{results['NNDSVD']['error']:.6f}",
                    delta=f"{results['NNDSVD']['error'] - results['Random']['error']:.6f}"
                )
            
            with col3:
                improvement = (norms[0] - norms[-1]) / norms[0] * 100
                st.metric(
                    "After MU", 
                    f"{results['Multiplicative Update']['error']:.6f}",
                    delta=f"-{improvement:.1f}% improvement"
                )
            
            # Convergence plot
            st.markdown('<h3 class="section-header">📉 Convergence Analysis</h3>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(norms, linewidth=2, color='#1f77b4')
            ax.set_title('NMF Convergence', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Frobenius Norm')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            st.pyplot(fig)
            
            # Matrix visualizations
            st.markdown('<h3 class="section-header">🎨 Matrix Visualizations</h3>', unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["W Matrix", "H Matrix", "Reconstruction", "Error Matrix"])
            
            with tab1:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # W matrices
                im1 = axes[0].imshow(results['Random']['W'], cmap='viridis', aspect='auto')
                axes[0].set_title('W (Random Init)')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].imshow(results['NNDSVD']['W'], cmap='viridis', aspect='auto')
                axes[1].set_title('W (NNDSVD Init)')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].imshow(results['Multiplicative Update']['W'], cmap='viridis', aspect='auto')
                axes[2].set_title('W (After MU)')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # H matrices
                im1 = axes[0].imshow(results['Random']['H'], cmap='viridis', aspect='auto')
                axes[0].set_title('H (Random Init)')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].imshow(results['NNDSVD']['H'], cmap='viridis', aspect='auto')
                axes[1].set_title('H (NNDSVD Init)')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].imshow(results['Multiplicative Update']['H'], cmap='viridis', aspect='auto')
                axes[2].set_title('H (After MU)')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab3:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original and reconstructions
                im1 = axes[0,0].imshow(A, cmap='viridis', aspect='auto')
                axes[0,0].set_title('Original Matrix A')
                plt.colorbar(im1, ax=axes[0,0])
                
                im2 = axes[0,1].imshow(results['Random']['reconstruction'], cmap='viridis', aspect='auto')
                axes[0,1].set_title('Reconstruction (Random)')
                plt.colorbar(im2, ax=axes[0,1])
                
                im3 = axes[1,0].imshow(results['NNDSVD']['reconstruction'], cmap='viridis', aspect='auto')
                axes[1,0].set_title('Reconstruction (NNDSVD)')
                plt.colorbar(im3, ax=axes[1,0])
                
                im4 = axes[1,1].imshow(results['Multiplicative Update']['reconstruction'], cmap='viridis', aspect='auto')
                axes[1,1].set_title('Reconstruction (After MU)')
                plt.colorbar(im4, ax=axes[1,1])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab4:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Error matrices
                error1 = A - results['Random']['reconstruction']
                error2 = A - results['NNDSVD']['reconstruction']
                error3 = A - results['Multiplicative Update']['reconstruction']
                
                im1 = axes[0].imshow(error1, cmap='RdBu', aspect='auto')
                axes[0].set_title('Error (Random)')
                plt.colorbar(im1, ax=axes[0])
                
                im2 = axes[1].imshow(error2, cmap='RdBu', aspect='auto')
                axes[1].set_title('Error (NNDSVD)')
                plt.colorbar(im2, ax=axes[1])
                
                im3 = axes[2].imshow(error3, cmap='RdBu', aspect='auto')
                axes[2].set_title('Error (After MU)')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Detailed statistics
            st.markdown('<h3 class="section-header">📊 Detailed Statistics</h3>', unsafe_allow_html=True)
            
            stats_data = []
            for method, result in results.items():
                if method != 'Multiplicative Update':
                    stats_data.append({
                        'Method': method,
                        'Reconstruction Error': f"{result['error']:.6f}",
                        'W Min': f"{result['W'].min():.3f}",
                        'W Max': f"{result['W'].max():.3f}",
                        'H Min': f"{result['H'].min():.3f}",
                        'H Max': f"{result['H'].max():.3f}"
                    })
                else:
                    stats_data.append({
                        'Method': method,
                        'Reconstruction Error': f"{result['error']:.6f}",
                        'W Min': f"{result['W'].min():.3f}",
                        'W Max': f"{result['W'].max():.3f}",
                        'H Min': f"{result['H'].min():.3f}",
                        'H Max': f"{result['H'].max():.3f}"
                    })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Convergence details
            if 'convergence' in results['Multiplicative Update']:
                st.markdown('<h3 class="section-header">🔍 Convergence Details</h3>', unsafe_allow_html=True)
                
                convergence_data = []
                norms = results['Multiplicative Update']['convergence']
                for i in range(0, len(norms), max(1, len(norms)//10)):
                    convergence_data.append({
                        'Iteration': i,
                        'Frobenius Norm': f"{norms[i]:.6f}",
                        'Improvement': f"{((norms[0] - norms[i])/norms[0]*100):.2f}%" if i > 0 else "0.00%"
                    })
                
                conv_df = pd.DataFrame(convergence_data)
                st.dataframe(conv_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔢 NMF Interactive Demo | Built with Streamlit</p>
    <p>Adjust parameters in the sidebar and click 'Run NMF Analysis' to see results</p>
</div>
""", unsafe_allow_html=True)