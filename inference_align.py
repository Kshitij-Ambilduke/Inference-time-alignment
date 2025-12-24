import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_feats import load_llama  # Assuming your previous code is saved here or integrated
import os
import json

# --- Configuration ---
SOURCE_LANG_CODE = "fra_Latn"   # Example: French
TARGET_LANG_CODE = "eng_Latn"   # English (for reference/sanity check)
MATRIX_PATH = "saved_feats_dir/alignment_matrices.pt"
ALPHA = 0.4
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
DEBUG_MODE = False 
OUTPUT_FOLDER = "data/inference_align_outputs" 

def get_intervention_hook(W, alpha, device, layer_idx):
    """
    Creates the hook function for a specific layer.
    """
    W = W.to(device).to(torch.float16)
    
    def hook_fn(module, args, output):
        hidden_states = output
        seq_len = hidden_states.shape[1]

        if seq_len > 1:
            # Assumes Left-Padding. 
            # If Right-Padding, this grabs a [PAD] token and wastes compute.
            h_last = hidden_states[:, -1, :] 
            
            projected = torch.matmul(h_last, W)
            
            # Apply intervention
            hidden_states[:, -1, :] = h_last + (alpha * projected)
            
        return hidden_states
        
    return hook_fn

def apply_incline(model, matrix_path="alignment_matrices.pt", alpha=0.4):
    matrices = torch.load(matrix_path)

    print(f"Applying INCLINE with alpha={alpha}...")

    # APPLY HOOK FOR EMBEDDING
    W = matrices[0]
    hook_fn = get_intervention_hook(W, alpha, model.device, 0)
    handle = model.model.embed_tokens.register_forward_hook(hook_fn)
    hooks = [handle]

    layers = model.model.layers
    
    for i, layer_module in enumerate(layers):
        # HF layers[i] -> FROM MODEL.MODEL.LAYERS
        #  output corresponds to 
        # hidden_states[i+1] -> FROM OUTPUT.RETURNED_HIDDEN_STATES
        # hidden_states[0] is embeddings WHICH IS SKIPPED HERE
        
        matrix_key = i + 1
        assert matrix_key>0, "Matrix key should be > 0 for layers"
        W = matrices[matrix_key]
        
        hook_fn = get_intervention_hook(W, alpha, model.device, i)
        handle = layer_module.register_forward_hook(hook_fn)
        hooks.append(handle)
            
    return hooks

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    # 'devtest' is the standard evaluation split (1012 sentences)
    # 'dev' is usually for validation (997 sentences)
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

if __name__ == "__main__":
    src_sentences = load_flores_data(SOURCE_LANG_CODE, split="devtest")
    ref_sentences = load_flores_data(TARGET_LANG_CODE, split="devtest")
    
    # DEBUG: Uncomment to run on a small subset first
    if DEBUG_MODE:
        src_sentences = src_sentences[:5]
        ref_sentences = ref_sentences[:5]

    # Load Model
    model, tokenizer = load_llama(quantized=True)
    
    # Setup Tokenizer for Left Padding
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    # Register Hooks
    active_hooks = apply_incline(model, MATRIX_PATH, alpha=ALPHA)

    # Processing Loop
    results = []
    print(f"Starting generation on {len(src_sentences)} sentences...")
    
    for i in tqdm(range(0, len(src_sentences), BATCH_SIZE)):
        batch_src = src_sentences[i : i + BATCH_SIZE]
        
        # Wrap in translation prompt (Adjust template as needed for Llama 3)
        # Simple template:
        prompts = [f"Translate to English: {s}" for s in batch_src]
        
        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, # Greedy decoding is preferred for translation accuracy
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode (skip the prompt)
        decoded_batch = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
        
        # Store/Print Results
        for src, gen, ref in zip(batch_src, decoded_batch, ref_sentences[i : i + BATCH_SIZE]):
            clean_gen = gen.strip()
            results.append({
                "source": src,
                "generated": clean_gen,
                "reference": ref
            })
            
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    output_path = os.path.join(OUTPUT_FOLDER, f"incline_alpha_{ALPHA}_{SOURCE_LANG_CODE}_to_{TARGET_LANG_CODE}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {output_path}")

    # Cleanup Hooks
    for handle in active_hooks:
        handle.remove()
    
    print("Processing complete.")