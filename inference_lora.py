'''
Inference on LoRA trained model
'''
import torch
import os
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel

# --- Constants ---
# Must match the training script exactly
LANG_ID_TO_NAME = {
    "eng_Latn": "English",
    "fra_Latn": "French",
    "spa_Latn": "Spanish",
    "swh_Latn": "Swahili",
    "npi_Deva": "Nepali",
    "deu_Latn": "German",
    "hin_Deva": "Hindi",
    "zho_Hans": "Chinese"
}

def load_flores_data(lang_code, split="devtest"):
    print(f"Loading FLORES {split} for {lang_code}...")
    ds = load_dataset("facebook/flores", lang_code, split=split, trust_remote_code=True)
    return ds['sentence']

def format_prompt(src_text, src_code):
    """
    Formats the input using the exact template from training.
    """
    src_lang_name = LANG_ID_TO_NAME.get(src_code, src_code)
    # Note: We end with "English:" just like the training labels started.
    return (
        f"Translate the following {src_lang_name} source sentence to English\n"
        f"Source: {src_text}\n"
        f"English:"
    )

def generate_translations(args):
    # 1. Configure Quantization
    if args.no_quant:
        print("Loading base model in bfloat16 (No Quantization)...")
        bnb_config = None
        torch_dtype = torch.bfloat16
    else:
        print("Loading base model in 4-bit NF4 (Quantized)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        torch_dtype = None

    # 2. Load Base Model
    print(f"Loading base model: {args.base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Load & Attach LoRA Adapter
    print(f"Loading LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval() # Set to evaluation mode

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    # Vital for generation: Pad on the LEFT so the prompt ends at the last position
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    # 5. Load Data
    src_sentences = load_flores_data(args.source_lang, split="devtest")
    ref_sentences = load_flores_data(args.target_lang, split="devtest")

    if args.debug:
        print("DEBUG MODE: Processing only 10 samples.")
        src_sentences = src_sentences[:10]
        ref_sentences = ref_sentences[:10]

    results = []
    print(f"Starting generation for {len(src_sentences)} samples...")

    # 6. Generation Loop
    for i in tqdm(range(0, len(src_sentences), args.batch_size)):
        batch_src = src_sentences[i : i + args.batch_size]
        
        # Create prompts
        prompts = [format_prompt(s, args.source_lang) for s in batch_src]
        
        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # Greedy decoding (deterministic)
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Optional: Stop repetition
                # repetition_penalty=1.1 
            )

        # Decode ONLY the new tokens (slice off the prompt)
        generated_tokens = output_ids[:, input_len:]
        decoded_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Save results
        current_refs = ref_sentences[i : i + args.batch_size]
        for src, gen, ref in zip(batch_src, decoded_batch, current_refs):
            results.append({
                "source": src,
                "generated": gen.strip(),
                "reference": ref
            })

    # 7. Save to File
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    filename = f"lora_{args.source_lang}_to_{args.target_lang}.jsonl"
    output_path = os.path.join(args.output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate translations using a trained LoRA adapter.")
    
    parser.add_argument("--base_model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the saved LoRA adapter folder (e.g. results/lora_baseline)")
    parser.add_argument("--source_lang", type=str, default="npi_Deva")
    parser.add_argument("--target_lang", type=str, default="eng_Latn")
    parser.add_argument("--output_dir", type=str, default="results/generation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_quant", action="store_true", help="Load model in BF16 instead of 4-bit")

    args = parser.parse_args()
    generate_translations(args)