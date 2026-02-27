'''
INCLINE Intervention generation
The same file can be used for Improved INCLINE
Since inference remains the same
'''

import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_hidden_features import load_llama   
import os
import json
import argparse

# --- Configuration ---
# SOURCE_LANG_CODE = "npi_Deva"   
# TARGET_LANG_CODE = "eng_Latn"   
# MATRIX_PATH = "/media/stoch-lab/Workspace/kshitij/nepali/flores_npi_Deva_to_eng_Latn_MATRICES_l2.pt" 
# ALPHA = 0.4 
# MAX_NEW_TOKENS = 64
# DEBUG_MODE = True 
# OUTPUT_FOLDER = "nepali" 

# --- Prompt Template Configuration ---
PROMPT_TEMPLATE = "Translate the following {src_lang} text to English.\nSource: {source_sentence}\nEnglish:"
SUFFIX_TEXT = "\nEnglish:" # The text that comes AFTER the source sentence

lang_id_to_name = {
    # --- Targets / Common ---
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "deu_Latn": "German",
    "hin_Deva": "Hindi",
    "zho_Hans": "Chinese",

    # --- Latin Script Group ---
    "swh_Latn": "Swahili",
    "lug_Latn": "Luganda",
    "som_Latn": "Somali",
    "ibo_Latn": "Igbo",
    "zul_Latn": "Zulu",

    # --- Devanagari Script Group ---
    "npi_Deva": "Nepali",
    "mai_Deva": "Maithili",
    "bho_Deva": "Bhojpuri",
    "san_Deva": "Sanskrit",
    "gom_Deva": "Konkani", 

    # --- Script Control Experiment ---
    # Both map to "Serbian" so the prompt reads "Translate the following Serbian text..."
    "srp_Cyrl": "Serbian",
    "srp_Latn": "Serbian",

    # --- Reverse Directionality ---
    "pbt_Arab": "Pashto",
    "urd_Arab": "Urdu",

    # --- Tokenization Density ---
    "mya_Mymr": "Burmese",
    "lao_Laoo": "Lao"
}

def parse_args():
    p = argparse.ArgumentParser(description="Run INCLINE inference with configurable options")
    p.add_argument("--source-lang", default="npi_Deva", help="FLORES source language code")
    p.add_argument("--target-lang", default="eng_Latn", help="FLORES target language code")
    p.add_argument("--matrix-path", default="/home/shishirk/adityasr/kshitij/results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt", help="Path to matrices .pt file")
    p.add_argument("--alpha", type=float, default=0.3, help="INCLINE alpha value")
    p.add_argument("--max-new-tokens", type=int, default=128, help="max_new_tokens for generation")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    p.add_argument("--output-folder", default="/home/shishirk/adityasr/kshitij/results/nepali_to_en", help="Output folder for results")
    p.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID for Llama")
    p.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization (load in full precision/fp16)")
    return p.parse_args()

def get_intervention_hook(W, alpha, device, layer_idx, target_idx):
    """
    Intervenes at the specific target_idx.
    In Batched Left-Padding, target_idx is consistent across the batch 
    relative to the tensor shape.
    """
    W = W.to(device).to(torch.float32) # Use float32 for stability
    
    def hook_fn(module, args, output):
        # 1. Handle Llama Tuple Output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # 2. Check if we are in the "Prefill" phase
        # seq_len will be > 1 during prefill (processing input prompt)
        seq_len = hidden_states.shape[1]
        
        if seq_len > 1:
            # 3. Validation: Ensure target index is within bounds
            if target_idx < seq_len:
                # Select the specific token across the ENTIRE batch
                # Shape: [Batch_Size, Hidden_Dim]
                h_source = hidden_states[:, target_idx, :] 
                
                # 4. Apply INCLINE Math
                h_source_f32 = h_source.to(torch.float32)
                projected = torch.matmul(h_source_f32, W)
                
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
    
    # Iterate through the transformer layers
    for i, layer_module in enumerate(model.model.layers):
        
        # Training Index i+1 matches output of layer i
        matrix_key = i + 1 
        
        if matrix_key in matrices:
            W = matrices[matrix_key]
            hook_fn = get_intervention_hook(W, alpha, model.device, i, target_token_index)
            handle = layer_module.register_forward_hook(hook_fn)
            hooks.append(handle)
        else:
            continue
            
    return hooks

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

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
    BATCH_SIZE = args.batch_size

    src_sentences = load_flores_data(SOURCE_LANG_CODE, split="devtest")
    ref_sentences = load_flores_data(TARGET_LANG_CODE, split="devtest")
    
    if DEBUG_MODE:
        src_sentences = src_sentences[:10]
        ref_sentences = ref_sentences[:10]
        BATCH_SIZE = 2 # Force small batch in debug

    model, tokenizer = load_llama(model_id=MODEL_ID, quantized=not args.no_quant)
    
    # CRITICAL FOR BATCH GENERATION: Left padding
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    # --- Pre-calculate suffix length ---
    suffix_tokens = tokenizer(SUFFIX_TEXT, add_special_tokens=False).input_ids
    suffix_len = len(suffix_tokens)
    print(f"Calculated suffix length ('{SUFFIX_TEXT}'): {suffix_len} tokens")

    results = []
    print(f"Starting INCLINE generation with Batch Size {BATCH_SIZE}...")
    
    # Combine src and ref to iterate together
    data_pairs = list(zip(src_sentences, ref_sentences))

    # Iterate in batches
    for batch in tqdm(batch_list(data_pairs, BATCH_SIZE), total=len(data_pairs)//BATCH_SIZE + 1):
        
        batch_src = [item[0] for item in batch]
        batch_ref = [item[1] for item in batch]

        # 1. Prepare Prompts
        batch_prompts = [
            PROMPT_TEMPLATE.format(
                src_lang=lang_id_to_name[SOURCE_LANG_CODE],
                source_sentence=s
            ) for s in batch_src
        ]
        
        # 2. Tokenize Batch (Left Padding is automatic due to tokenizer config above)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        # 3. Calculate Intervention Index
        # With Left Padding: [PAD, PAD, Start, ..., EndSource, Suffix1, Suffix2]
        # The sequence length is 'input_length'.
        # The Suffix is always at the very end of the sequence.
        # Therefore, EndSource is always at: input_length - suffix_len - 1
        intervention_idx = input_length - suffix_len - 1
        
        # 4. Register Hooks
        active_hooks = apply_incline(model, MATRIX_PATH, ALPHA, intervention_idx)

        # 5. Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 6. Remove Hooks
        for h in active_hooks: h.remove()
        
        # 7. Decode Batch
        # We slice [:, input_length:] to get only the generated tokens
        generated_texts = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
        
        for src, ref, gen in zip(batch_src, batch_ref, generated_texts):
            results.append({
                "source": src,
                "generated": gen.strip(),
                "reference": ref
            })

    # Save
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    output_path = os.path.join(OUTPUT_FOLDER, f"incline_{SOURCE_LANG_CODE}_alpha{ALPHA}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {output_path}")