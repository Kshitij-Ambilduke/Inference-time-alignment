import torch
from extract_feats import load_llama

def get_intervention_hook(W, alpha, device, layer_idx):
    """
    Creates the hook function for a specific layer.
    """
    W = W.to(device).to(torch.float16)
    
    def hook_fn(module, args, output):
        # output[0] is hidden_states: [batch_size, seq_len, hidden_dim]
        # print(output.shape)
        hidden_states = output
        
        # We need the attention mask to find the last real token.
        # Unfortunately, forward hooks on layers don't receive the mask directly in 'args'.
        # We must rely on the mask being passed implicitly or hack it. 
        # Ideally, we pass the mask to this function generator or rely on left-padding.
        
        # ROBUST FIX FOR PADDING:
        # Since we cannot easily access attention_mask inside a standard nn.Module forward hook
        # without complex partials, we enforce a constraint:
        # We assume the user provides inputs that are LEFT-PADDED (standard for generation).
        # In left-padding, the last token IS the last index [-1].
        
        # However, if we want to be 100% robust against right-padding, we'd need to 
        # capture the mask in the outer scope. For this snippet, we will stick to [-1]
        # but add a warning that inputs MUST be left-padded.
        
        # If you strictly need mask support, you'd calculate indices outside and pass them in,
        # but that breaks the stateless hook signature. 
        # print(hidden_states.shape)
        seq_len = hidden_states.shape[1]

        # 1. PREFILL (Processing Prompt)
        if seq_len > 1:
            # Assumes Left-Padding. 
            # If Right-Padding, this grabs a [PAD] token and wastes compute.
            h_last = hidden_states[:, -1, :] 
            
            projected = torch.matmul(h_last, W)
            
            # Apply intervention
            # Using in-place modification on the slice
            hidden_states[:, -1, :] = h_last + (alpha * projected)
            
        return hidden_states
        
    return hook_fn

def apply_incline(model, matrix_path="alignment_matrices.pt", alpha=0.4):
    matrices = torch.load(matrix_path)
    layers = model.model.layers
    hooks = []
    
    print(f"Applying INCLINE with alpha={alpha}...")
    
    for i, layer_module in enumerate(layers):
        # FIX: Off-by-one error.
        # HF layers[i] output corresponds to hidden_states[i+1]
        # hidden_states[0] is embeddings (which we skip here as we can't easily hook embeddings output directly via model.layers)
        
        matrix_key = i + 1
        
    # if matrix_key in matrices:
            # We use the matrix trained on Layer i's OUTPUT

        W = matrices[matrix_key]
        
        hook_fn = get_intervention_hook(W, alpha, model.device, i)
        handle = layer_module.register_forward_hook(hook_fn)
        hooks.append(handle)
            
    return hooks

# --- Main Execution ---

# 1. Load Model
model, tokenizer = load_llama(quantized=True)

# Important: Set tokenizer to Left Padding for generation correctness
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

# 2. Register Hooks
active_hooks = apply_incline(model, "saved_feats_dir/alignment_matrices.pt", alpha=0.4)

# 3. Prepare Input
input_text = ["Translate to English: El precio del oro ha subido mucho.", "Translate to English: El precio del oro ha subido mucho."]
# Ensure we pad if batching, though here batch=1
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

# 4. Generate
print(f"\nPrompt: {input_text[0]}")
with torch.no_grad():
    output_ids = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# 5. Decode
input_length = inputs.input_ids.shape[1]
generated_text = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True)
print(f"Output: {generated_text}")

# 6. Cleanup
for handle in active_hooks:
    handle.remove()