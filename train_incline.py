import torch
import argparse
import os
import sys

def ridge_dual(X, Y, lam=1e-2):
    """
    Computes the Ridge Regression solution using the dual formulation.
    Ideally faster when N < d (number of samples < hidden dimension).
    
    Args:
        X (Tensor): Source data [N, d]
        Y (Tensor): Target data [N, d]
        lam (float): Regularization strength
    Returns:
        W (Tensor): Weight matrix [d, d] such that XW ~ Y
    """
    # X,Y: (N,d)
    N = X.shape[0]
    
    # Compute Gram matrix XX^T: (N, N)
    XXt = X @ X.T
    
    # Regularize: (XX^T + lambda * I)
    A = XXt + lam * torch.eye(N, device=X.device, dtype=X.dtype)
    
    # Solve A * Z = Y for Z.  (Z will be (N, d))
    # We use linalg.solve because A is positive definite
    AinvY = torch.linalg.solve(A, Y)
    
    # Recover Primal W: W = X^T * Z
    W = X.T @ AinvY  # (d, d)
    
    return W

def train_alignment_matrices(input_file, output_file, lam=1e-2):
    print(f"Loading features from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    data = torch.load(input_file)
    H_s_dict = data["source"]
    H_t_dict = data["target"]
    
    alignment_matrices = {}
    num_layers = len(H_s_dict)
    
    # Check for CUDA to speed up matrix math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training matrices for {num_layers} layers (Ridge Dual, lambda={lam})...")
    
    for layer_idx in sorted(H_s_dict.keys()):
        # X and Y shape: [num_samples, hidden_dim]
        # We cast to double (float64) for numerical stability during matrix inversion
        X = H_s_dict[layer_idx].double().to(device) 
        Y = H_t_dict[layer_idx].double().to(device)
        
        # X = H_s_dict[layer_idx].to(device) 
        # Y = H_t_dict[layer_idx].to(device)

        W = ridge_dual(X, Y, lam=lam) # Shape [hidden_dim, hidden_dim]

        # Save back to CPU to save memory/disk space
        alignment_matrices[layer_idx] = W.cpu()
        
        # Optional: Print simple progress
        if layer_idx % 5 == 0:
            print(f"  Processed layer {layer_idx}")

    # Ensure directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(alignment_matrices, output_file)
    print(f"Alignment matrices trained and saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train alignment matrices using Ridge Regression (Dual Form).")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the .pt file containing extracted hidden states (source and target)."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True, 
        help="Path to save the resulting alignment matrices."
    )
    
    parser.add_argument(
        "--lam", 
        type=float, 
        default=1e-2, 
        help="Regularization parameter (lambda) for Ridge Regression. Default: 0.01"
    )

    args = parser.parse_args()

    train_alignment_matrices(
        input_file=args.input_file, 
        output_file=args.output_file, 
        lam=args.lam
    )