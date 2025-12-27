import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_hidden_features import load_llama  
import os
import json
import argparse

# # --- Configuration ---
# SOURCE_LANG_CODE = "npi_Deva"   
# TARGET_LANG_CODE = "eng_Latn"   
# MATRIX_PATH = "/media/stoch-lab/Workspace/kshitij/nepali/flores_npi_Deva_to_eng_Latn_MATRICES_l2.pt" # Update to your actual path
# ALPHA = 0.4 # Start conservative. Paper suggests range -1 to 1.
# MAX_NEW_TOKENS = 64
# DEBUG_MODE = True 
# OUTPUT_FOLDER = "nepali" 

# --- Prompt Template Configuration ---
# We calculate the suffix length to find the correct intervention index
PROMPT_TEMPLATE = "Translate the following {src_lang} text to English.\nSource: {source_sentence}\nEnglish:"
SUFFIX_TEXT = "\nEnglish:" # The text that comes AFTER the source sentence

lang_id_to_name = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "swh_Latn": "Swahili",
    "npi_Deva": "Nepali"
}

def parse_args():
    p = argparse.ArgumentParser(description="Run INCLINE inference with configurable options")
    p.add_argument("--source-lang", default="npi_Deva", help="FLORES source language code")
    p.add_argument("--target-lang", default="eng_Latn", help="FLORES target language code")
    p.add_argument("--matrix-path", default="/home/shishirk/adityasr/kshitij/results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt", help="Path to matrices .pt file")
    p.add_argument("--alpha", type=float, default=0.3, help="INCLINE alpha value")
    p.add_argument("--max-new-tokens", type=int, default=128, help="max_new_tokens for generation")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    p.add_argument("--output-folder", default="/home/shishirk/adityasr/kshitij/results/nepali_to_en", help="Output folder for results")
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID for Llama")
    return p.parse_args()

def get_intervention_hook(W, alpha, device, layer_idx, target_idx):
    """
    Intervenes ONLY at the specific target_idx (end of source sentence).
    """
    W = W.to(device).to(torch.float32) # Use float32 for stability
    
    def hook_fn(module, args, output):
        # 1. Handle Llama Tuple Output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # 2. Check if we are in the "Prefill" phase (processing the prompt)
        # If seq_len == 1, we are generating new tokens (do not intervene)
        seq_len = hidden_states.shape[1]
        if seq_len > 1:
            
            # 3. Validation: Ensure target index is within bounds
            if target_idx < seq_len:
                h_source = hidden_states[:, target_idx, :] # Shape: [Batch, Hidden]
                
                # 4. Apply INCLINE Math
                # Cast to float32 for matmul to avoid float16 overflow
                h_source_f32 = h_source.to(torch.float32)
                projected = torch.matmul(h_source_f32, W)
                
                # Debug print (first layer only) to check for explosion
                if layer_idx == 0 and DEBUG_MODE:
                    mag_orig = torch.norm(h_source_f32).item()
                    mag_proj = torch.norm(projected).item()
                    if mag_proj > mag_orig * 10:
                        print(f"WARNING: Layer 0 projection explosion! Orig: {mag_orig:.1f}, Proj: {mag_proj:.1f}")

                intervention_vector = alpha * projected
                
                # Inject back
                hidden_states[:, target_idx, :] = h_source + intervention_vector.to(hidden_states.dtype)

        # 5. Return correct format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
        
    return hook_fn

def apply_incline(model, matrix_path, alpha, target_token_index):
    matrices = torch.load(matrix_path)
    hooks = []
    
    # Iterate through the transformer layers (0 to 31 for Llama-3-8B)
    for i, layer_module in enumerate(model.model.layers):
        
        # --- CORRECTION ---
        # Training Index 0 = Embeddings
        # Training Index 1 = Output of Layer 0
        # ...
        # Since we are hooking the output of Layer 'i', we need Training Index 'i+1'
        matrix_key = i + 1 
        
        if matrix_key in matrices:
            W = matrices[matrix_key]
            
            # Pass i (layer_idx) for debugging print statements if needed
            hook_fn = get_intervention_hook(W, alpha, model.device, i, target_token_index)
            handle = layer_module.register_forward_hook(hook_fn)
            hooks.append(handle)
        else:
            # If the matrix is missing, we just skip this layer
            continue
            
    return hooks

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

if __name__ == "__main__":
    args = parse_args()

    SOURCE_LANG_CODE = args.source_lang
    TARGET_LANG_CODE = args.target_lang
    MATRIX_PATH = args.matrix_path
    ALPHA = args.alpha
    MAX_NEW_TOKENS = args.max_new_tokens
    DEBUG_MODE = args.debug
    OUTPUT_FOLDER = args.output_folder
    MODEL_ID = args.model_id

    src_sentences = load_flores_data(SOURCE_LANG_CODE, split="devtest")
    ref_sentences = load_flores_data(TARGET_LANG_CODE, split="devtest")
    
    if DEBUG_MODE:
        src_sentences = src_sentences[:20]
        ref_sentences = ref_sentences[:20]

    model, tokenizer = load_llama(model_id=MODEL_ID,quantized=True)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    # --- Pre-calculate suffix length for alignment ---
    # We need to know where the source sentence ends in the token sequence.
    # Prompt: "... Source: {SENTENCE}\nEnglish:"
    # The intervention must happen at the last token of {SENTENCE}.
    # This is exactly (Total_Len - Suffix_Len - 1)
    suffix_tokens = tokenizer(SUFFIX_TEXT, add_special_tokens=False).input_ids
    suffix_len = len(suffix_tokens)
    print(f"Calculated suffix length ('{SUFFIX_TEXT}'): {suffix_len} tokens")

    results = []
    print(f"Starting INCLINE generation...")
    
    # We use Batch Size = 1 to guarantee index correctness
    for src, ref in tqdm(zip(src_sentences, ref_sentences), total=len(src_sentences)):
        
        # 1. Prepare Prompt
        prompt = PROMPT_TEMPLATE.format(
            src_lang=lang_id_to_name[SOURCE_LANG_CODE],
            source_sentence=src
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        seq_len = inputs.input_ids.shape[1]
        
        # 2. Calculate Intervention Index
        # We target the token BEFORE the suffix starts.
        # Index = Total_Len - Suffix_Len - 1
        intervention_idx = seq_len - suffix_len - 1
        
        # 3. Register Hooks for this specific index
        active_hooks = apply_incline(model, MATRIX_PATH, ALPHA, intervention_idx)

        # 4. Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 5. Remove Hooks immediately
        for h in active_hooks: h.remove()
        
        # 6. Decode
        gen_text = tokenizer.decode(output_ids[0, seq_len:], skip_special_tokens=True)
        
        results.append({
            "source": src,
            "generated": gen_text.strip(),
            "reference": ref
        })

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    output_path = os.path.join(OUTPUT_FOLDER, f"incline_{SOURCE_LANG_CODE}_alpha{ALPHA}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {output_path}")