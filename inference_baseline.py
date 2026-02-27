'''
For generating baseline results
Without INCLINE or Improved INCLINE
Normal Base mode generation
'''
import torch
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import argparse
import sys
from extract_hidden_features import load_llama

# --- Constants ---
LANG_ID_TO_NAME = {
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

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

def parse_args():
    parser = argparse.ArgumentParser(description="Run Baseline Llama generation (No Intervention).")
    
    parser.add_argument("--source_lang", type=str, default="npi_Deva", help="Source language FLORES code")
    parser.add_argument("--target_lang", type=str, default="eng_Latn", help="Target language FLORES code")
    
    parser.add_argument("--output_dir", type=str, default="nepali", help="Folder to save results")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF Model ID")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    
    parser.add_argument("--debug", action="store_true", help="Run on a small subset (20 samples) for debugging")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load Data
    src_sentences = load_flores_data(args.source_lang, split="devtest")
    ref_sentences = load_flores_data(args.target_lang, split="devtest")
    
    if args.debug:
        print("DEBUG MODE: Processing only 20 samples.")
        src_sentences = src_sentences[:20]
        ref_sentences = ref_sentences[:20]

    # Load Model
    model, tokenizer = load_llama(args.model_id, quantized=not args.no_quant)
    
    # Configure Tokenizer for Batch Generation (Left Padding is crucial)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    # Determine Language Name for Prompt
    src_lang_name = LANG_ID_TO_NAME.get(args.source_lang, args.source_lang)

    results = []
    print(f"Starting BASELINE generation on {len(src_sentences)} sentences...")
    
    # Batch Processing Loop
    for i in tqdm(range(0, len(src_sentences), args.batch_size)):
        batch_src = src_sentences[i : i + args.batch_size]
        
        # Dynamic Prompt Construction
        prompts = [
            f"Translate the following {src_lang_name} source sentence to English\nSource: {s}\nEnglish:" 
            for s in batch_src
        ]
        
        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        decoded_batch = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
        
        # Store Results
        current_refs = ref_sentences[i : i + args.batch_size]
        for src, gen, ref in zip(batch_src, decoded_batch, current_refs):
            results.append({
                "source": src,
                "generated": gen.strip(),
                "reference": ref
            })

    # Save Results
    filename = f"baseline_{args.source_lang}_to_{args.target_lang}.jsonl"
    output_path = os.path.join(args.output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in results:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Baseline results saved to {output_path}")
    print("Processing complete.")