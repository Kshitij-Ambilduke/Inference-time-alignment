import torch
from datasets import load_dataset
from tqdm import tqdm
from extract_feats import load_llama
import os
import json

# --- Configuration ---
SOURCE_LANG_CODE = "fra_Latn"   # Example: French
TARGET_LANG_CODE = "eng_Latn"   # English
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
DEBUG_MODE = False 
OUTPUT_FOLDER = "data/inference_align_outputs" 

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

if __name__ == "__main__":
    src_sentences = load_flores_data(SOURCE_LANG_CODE, split="devtest")
    ref_sentences = load_flores_data(TARGET_LANG_CODE, split="devtest")
    
    if DEBUG_MODE:
        src_sentences = src_sentences[:5]
        ref_sentences = ref_sentences[:5]

    model, tokenizer = load_llama(quantized=True)
    
    # tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    print(f"Starting BASELINE generation on {len(src_sentences)} sentences...")
    
    for i in tqdm(range(0, len(src_sentences), BATCH_SIZE)):
        batch_src = src_sentences[i : i + BATCH_SIZE]
        
        # Use exact same prompt template as intervention script
        prompts = [f"Translate to English: {s}" for s in batch_src]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        decoded_batch = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
        
        for src, gen, ref in zip(batch_src, decoded_batch, ref_sentences[i : i + BATCH_SIZE]):
            clean_gen = gen.strip()
            results.append({
                "source": src,
                "generated": clean_gen,
                "reference": ref
            })

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # Naming convention: baseline_{src}_to_{tgt}.jsonl
    output_path = os.path.join(OUTPUT_FOLDER, f"baseline_{SOURCE_LANG_CODE}_to_{TARGET_LANG_CODE}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Baseline results saved to {output_path}")
    print("Processing complete.")