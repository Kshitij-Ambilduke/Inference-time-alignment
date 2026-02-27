'''
Extracting hidden features of the last token
For a given pair of languages
Which are to be used for training 
the alignment matrices
'''
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset
import os
import random
import argparse

# TO CHECK -> PADDING SIDE OF LLAMA TOKENIZER

def load_flores_pair(src_lang, tgt_lang="eng_Latn", split="dev", eggs=500):
    """
    Loads parallel sentences from the FLORES dataset.
    """
    print(f"Loading FLORES dataset: {src_lang} -> {tgt_lang} ({split})")
    
    # Load Source
    ds_src = load_dataset("facebook/flores", src_lang, split=split, trust_remote_code=True)
    # Load Target
    ds_tgt = load_dataset("facebook/flores", tgt_lang, split=split, trust_remote_code=True)
    
    # Extract sentences list
    src_sentences = ds_src['sentence']
    tgt_sentences = ds_tgt['sentence']
    
    # Verify alignment
    assert len(src_sentences) == len(tgt_sentences), "Datasets are not perfectly aligned!"
    
    # shuffle and take eggs samples
    combined = list(zip(src_sentences, tgt_sentences))
    random.shuffle(combined)
    src_sentences, tgt_sentences = zip(*combined)
    
    # Slice only if we have more sentences than requested
    if eggs < len(src_sentences):
        src_sentences = src_sentences[:eggs]
        tgt_sentences = tgt_sentences[:eggs]

    print(f"Loaded {len(src_sentences)} parallel pairs.")
    
    return src_sentences, tgt_sentences

def load_llama(model_id, quantized=True):
    if quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 
    # Important for generation tasks, but strictly for feature extraction of the *last* token, 
    # right padding is usually standard/easier to index (-1).
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer

def get_layer_hidden_states(model, tokenizer, sentences, batch_size=4):
    """
    Extracts the last-token hidden states for every layer.
    Returns: Dictionary {layer_idx: Tensor(num_samples, hidden_dim)}
    """
    all_layer_states = {i: [] for i in range(model.config.num_hidden_layers + 1)} # +1 for embeddings layer

    for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting features"):
        batch_sentences = sentences[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Last token's embedding for each sentence in the batch        
        last_token_indices = inputs.attention_mask.sum(1) - 1 
        
        for layer_idx, layer_state in enumerate(outputs.hidden_states):
            # layer_state shape: [batch, seq_len, hidden_dim]
            batch_last_states = layer_state[torch.arange(layer_state.shape[0]), last_token_indices]  
            # batch_last_states shape: [batch, hidden_dim]          
            all_layer_states[layer_idx].append(batch_last_states.cpu())
    
    # Concatenate all batches
    for layer_idx in all_layer_states:
        all_layer_states[layer_idx] = torch.cat(all_layer_states[layer_idx], dim=0)
        
    return all_layer_states

def parse_args():
    parser = argparse.ArgumentParser(description="Extract hidden states from Llama models using FLORES dataset.")
    
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code (e.g., npi_Deva, fra_Latn)")
    parser.add_argument("--target_lang", type=str, default="eng_Latn", help="Target language code (default: eng_Latn)")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--save_dir", type=str, default="results/nepali_to_en", help="Directory to save the results")
    parser.add_argument("--samples", type=int, default=500, help="Number of sentence pairs to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for feature extraction")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer samples")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization (load in full precision/fp16)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Adjust samples for debug mode
    samples_to_load = 10 if args.debug else args.samples
    if args.debug:
        print("DEBUG MODE ENABLED: Using only 10 samples.")

    source_sentences, target_sentences = load_flores_pair(
        src_lang=args.source_lang, 
        tgt_lang=args.target_lang, 
        eggs=samples_to_load
    )

    model, tokenizer = load_llama(
        model_id=args.model_id, 
        quantized=not args.no_quant
    )

    print(f"Extracting features for {args.source_lang}...")
    H_source = get_layer_hidden_states(
        model, 
        tokenizer, 
        source_sentences, 
        batch_size=args.batch_size
    )

    print(f"Extracting features for {args.target_lang}...")
    H_target = get_layer_hidden_states(
        model, 
        tokenizer, 
        target_sentences, 
        batch_size=args.batch_size
    )

    filename = f"flores_{args.source_lang}_to_{args.target_lang}.pt"
    save_path = os.path.join(args.save_dir, filename)
    
    torch.save({"source": H_source, "target": H_target}, save_path)
    print(f"Features saved to {save_path}")