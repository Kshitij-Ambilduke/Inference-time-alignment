import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_hidden_features import load_llama
import os
import random
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run cascading layer-wise alignment training.")
    
    # Path Arguments
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output .pt file")
    parser.add_argument("--old_features_file", type=str, required=True, help="Path to the pre-extracted target (English) features .pt file")
    
    # Model & Data Arguments
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--source_lang", type=str, default="swh_Latn", help="FLORES source language code")
    parser.add_argument("--target_lang", type=str, default="eng_Latn", help="FLORES target language code")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of random samples to use for training")
    
    # Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.4, help="Intervention strength (alpha)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--lambda_reg", type=float, default=1e-2, help="Ridge regression regularization strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

def ridge_dual(X, Y, lam=1e-2):
    # Solves W such that X @ W ≈ Y
    N = X.shape[0]
    XXt = X @ X.T
    A = XXt + lam * torch.eye(N, device=X.device, dtype=X.dtype)
    AinvY = torch.linalg.solve(A, Y)
    W = X.T @ AinvY
    return W

def get_layer_outputs(model, tokenizer, sentences, batch_size=4, target_layer_idx=0):
    """
    Runs the model and extracts JUST the output of 'target_layer_idx'.
    Crucially, because hooks are registered on the model, 
    this forward pass includes ALL previous interventions!
    """
    layer_outputs = []
    
    # Simple progress bar for batches
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Layer {target_layer_idx} extraction", leave=False):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Target Layer Index mapping:
        # hidden_states[0] = Embeddings
        # hidden_states[1] = Output of Layer 0
        # ...
        # hidden_states[i+1] = Output of Layer i
        
        # We want the output of the specific layer index
        hidden_idx = target_layer_idx + 1
        
        # Extract last token (handling left padding if necessary, though llama usually rights pads)
        # Note: If your tokenizer uses left padding, -1 is correct. If right padding, ensure attention_mask usage or explicit index.
        # Assuming standard Llama right-padding behavior or unpadded specific tokens:
        batch_last_states = outputs.hidden_states[hidden_idx][:, -1, :]
        layer_outputs.append(batch_last_states)
        
    return torch.cat(layer_outputs, dim=0) # [N, hidden_dim]

def get_intervention_hook(W, alpha):
    # Standard hook to apply the intervention PERMANENTLY during training
    # Ensure W is on the correct device when the hook fires
    
    def hook_fn(module, args, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        
        # Move W to the device of the input tensor dynamically to avoid device mismatches
        W_device = W.to(h.device).float()
        
        # Apply to ALL tokens because we need the context to drift correctly
        # for the next layer's calculation.
        h_f32 = h.float()
        proj = torch.matmul(h_f32, W_device)
        intervention = alpha * proj
        
        # Inject
        h_new = h + intervention.to(h.dtype)
        
        if isinstance(output, tuple): return (h_new,) + output[1:]
        return h_new
    return hook_fn

# --- Main Cascading Loop ---
if __name__ == "__main__":
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data & Model
    print(f"Loading Data (Source: {args.source_lang}, Target: {args.target_lang})...")
    ds_src = load_dataset("facebook/flores", args.source_lang, split="dev")['sentence']
    ds_tgt = load_dataset("facebook/flores", args.target_lang, split="dev")['sentence']

    # Select random samples for training
    random.seed(args.seed)
    if len(ds_src) < args.num_samples:
        print(f"Warning: requested {args.num_samples} samples but dataset only has {len(ds_src)}. Using full dataset.")
        selected_indices = range(len(ds_src))
    else:
        selected_indices = random.sample(range(len(ds_src)), args.num_samples)
    
    ds_src = [ds_src[i] for i in selected_indices]
    ds_tgt = [ds_tgt[i] for i in selected_indices]

    print(f"Loading Model: {args.model_id}...")
    # Ensure hooks can be registered (quantized=True usually works with bitsandbytes)
    model, tokenizer = load_llama(model_id=args.model_id, quantized=True) 
    
    # 2. Load Pre-extracted Targets
    print(f"Loading pre-extracted targets from {args.old_features_file}...")
    if not os.path.exists(args.old_features_file):
        print(f"Error: Old features file not found at {args.old_features_file}")
        sys.exit(1)
        
    old_data = torch.load(args.old_features_file)
    # Check if key is 'target' or 'english' or just the dict, adapt based on your specific file structure
    # Assuming structure from prompt:
    H_t_dict = old_data["target"]

    alignment_matrices = {}
    active_hooks = []
    
    # 3. Iterate Layers
    num_layers = len(model.model.layers)
    print(f"Starting Cascading Training for {num_layers} layers with Alpha={args.alpha}...")
    
    for layer_idx in range(num_layers):
        print(f"--- Processing Layer {layer_idx} ---")
        
        # A. Extract Source Features for this layer
        # Since 'active_hooks' are registered, this forward pass 
        # includes the cumulative effect of layers 0 to layer_idx-1
        X = get_layer_outputs(model, tokenizer, ds_src, args.batch_size, layer_idx)
        X = X.double().cuda()
        
        # B. Get Target Features
        # Ensure we have data for this layer
        if layer_idx not in H_t_dict:
            print(f"Warning: No target data found for layer {layer_idx}. Skipping or stopping.")
            break

        Y = H_t_dict[layer_idx].double().cuda()
        
        # C. Train Matrix W
        # Map (Drifted Source) -> (Clean Target)
        W = ridge_dual(X, Y, lam=args.lambda_reg)
        alignment_matrices[layer_idx + 1] = W.cpu()
        
        # D. REGISTER HOOK immediately
        # This ensures the next iteration (Layer + 1) sees the intervention from Layer
        layer_module = model.model.layers[layer_idx]
        hook = get_intervention_hook(W, args.alpha)
        handle = layer_module.register_forward_hook(hook)
        active_hooks.append(handle)
        
        # Clean up GPU
        del X, Y, W
        torch.cuda.empty_cache()

    # 4. Save
    print(f"Saving alignment matrices to {args.output_file}...")
    torch.save(alignment_matrices, args.output_file)
    print("Cascading Training Complete. Hooks removed (process ending).")