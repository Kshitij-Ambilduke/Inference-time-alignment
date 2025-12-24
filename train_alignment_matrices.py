import torch

def train_alignment_matrices(feature_file="extracted_features.pt"):
    data = torch.load(feature_file)
    H_s_dict = data["source"]
    H_t_dict = data["target"]
    
    alignment_matrices = {}
    num_layers = len(H_s_dict)
    
    print(f"Training matrices for {num_layers} layers...")
    
    for layer_idx in range(num_layers):
        # Shape: [num_samples, hidden_dim]
        X = H_s_dict[layer_idx].float() # Source
        Y = H_t_dict[layer_idx].float() # Target
        
        # Solve Y = XW => W = lstsq(X, Y)
        # torch.linalg.lstsq returns a solution that minimizes ||AX - B||
        # Here we want XW ~ Y. 
        # Note: lstsq expects shape (samples, features). 
        
        # Check for CUDA to speed up matrix math
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = X.to(device)
        Y = Y.to(device)
        
        # Driver 'gels' is standard for least squares
        result = torch.linalg.lstsq(X, Y, driver="gels")
        W = result.solution # Shape [hidden_dim, hidden_dim]
        
        alignment_matrices[layer_idx] = W.cpu()
        
        # Optional: Print residual error to track quality
        # error = torch.mean((X @ W - Y)**2)
        # print(f"Layer {layer_idx} MSE: {error.item():.4f}")

    torch.save(alignment_matrices, "saved_feats_dir/alignment_matrices.pt")
    print("Alignment matrices trained and saved.")

# Run training
train_alignment_matrices(feature_file="saved_feats_dir/extracted_features.pt")