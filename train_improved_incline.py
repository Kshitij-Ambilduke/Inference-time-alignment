import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_hidden_states import load_llama
import os

# --- Config ---
SOURCE_LANG_CODE = "npi_Deva"
TARGET_LANG_CODE = "eng_Latn"
ALPHA = 0.4 
OUTPUT_FILE = "/media/stoch-lab/Workspace/kshitij/nepali/flores_npi_Deva_to_eng_Latn_cascading_alignment_matrices.pt"
BATCH_SIZE = 4
LAMBDA_REG = 1e-2
OLD_FEATURES_FILE = "/media/stoch-lab/Workspace/kshitij/nepali/flores_npi_Deva_to_eng_Latn.pt"

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
    
    for i in range(0, len(sentences), batch_size):
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
        
        # Extract last token (handling left padding)
        batch_last_states = outputs.hidden_states[hidden_idx][:, -1, :]
        layer_outputs.append(batch_last_states)
        
    return torch.cat(layer_outputs, dim=0) # [N, hidden_dim]

def get_intervention_hook(W, alpha):
    # Standard hook to apply the intervention PERMANENTLY during training
    W = W.to("cuda" if torch.cuda.is_available() else "cpu").float()
    
    def hook_fn(module, args, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        
        # Apply to ALL tokens because we need the context to drift correctly
        # for the next layer's calculation.
        h_f32 = h.float()
        proj = torch.matmul(h_f32, W)
        intervention = alpha * proj
        
        # Inject
        h_new = h + intervention.to(h.dtype)
        
        if isinstance(output, tuple): return (h_new,) + output[1:]
        return h_new
    return hook_fn

# --- Main Cascading Loop ---
if __name__ == "__main__":
    # 1. Load Data & Model
    print("Loading Data...")
    ds_src = load_dataset("facebook/flores", SOURCE_LANG_CODE, split="devtest")['sentence'][:500]
    ds_tgt = load_dataset("facebook/flores", TARGET_LANG_CODE, split="devtest")['sentence'][:500]
    
    model, tokenizer = load_llama(quantized=True) # Ensure hooks can be registered
    
    # 2. Extract TARGET (English) features once (they don't change)
    print("Pre-extracting English Targets...")
    # This is still fast because we don't intervene on English
    # You can reuse your old extraction code for this part or run it here.
    # For simplicity, let's assume you have H_t_dict loaded from your old .pt file:
    old_data = torch.load(OLD_FEATURES_FILE)
    H_t_dict = old_data["target"] # Dictionary of tensors
    
    alignment_matrices = {}
    active_hooks = []
    
    # 3. Iterate Layers
    num_layers = len(model.model.layers)
    print(f"Starting Cascading Training for {num_layers} layers...")
    
    for layer_idx in range(num_layers):
        print(f"--- Processing Layer {layer_idx} ---")
        
        # A. Extract Source Features for this layer
        # Since 'active_hooks' are registered, this forward pass 
        # includes the cumulative effect of layers 0 to layer_idx-1
        X = get_layer_outputs(model, tokenizer, ds_src, BATCH_SIZE, layer_idx)
        X = X.double().cuda()
        
        # B. Get Target Features
        Y = H_t_dict[layer_idx].double().cuda()
        
        # C. Train Matrix W
        # Map (Drifted Source) -> (Clean Target)
        W = ridge_dual(X, Y, lam=LAMBDA_REG)
        alignment_matrices[layer_idx + 1] = W.cpu()
        
        # D. REGISTER HOOK immediately
        # This ensures the next iteration (Layer + 1) sees the intervention from Layer
        layer_module = model.model.layers[layer_idx]
        hook = get_intervention_hook(W, ALPHA)
        handle = layer_module.register_forward_hook(hook)
        active_hooks.append(handle)
        
        # Clean up GPU
        del X, Y, W
        torch.cuda.empty_cache()

    # 4. Save
    torch.save(alignment_matrices, OUTPUT_FILE)
    print("Cascading Training Complete. Hooks removed.")