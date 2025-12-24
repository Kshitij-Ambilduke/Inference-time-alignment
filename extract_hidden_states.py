import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datasets import load_dataset
import os


# TO CHECK -> PADDING SIDE OF LLAMA TOKENIZER

# --- Configuration ---
SOURCE_LANG_CODE = "fra_Latn"  # Example: French. Change this to your 'xx' language
TARGET_LANG_CODE = "eng_Latn"  # English
Model_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR = "data"
DEBUG_MODE = False  # Set to True for quick testing with fewer samples

def load_flores_pair(src_lang, tgt_lang="eng_Latn", split="dev"):
    """
    Loads parallel sentences from the FLORES dataset.
    FLORES codes examples: 
    - French: fra_Latn
    - Spanish: spa_Latn
    - Swahili: swh_Latn
    - Nepali: npi_Deva
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
    print(f"Loaded {len(src_sentences)} parallel pairs.")
    
    return src_sentences, tgt_sentences

def load_llama(quantized=True, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    model_id = model_id    
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
    # print(all_layer_states.__len__())

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
            print(outputs.hidden_states.__len__())
            print(outputs.hidden_states[0].shape)
        
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
    # all_layer_states[layer_idx].shape: [num_samples, hidden_dim] 
        
    return all_layer_states

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    source_sentences, target_sentences = load_flores_pair(SOURCE_LANG_CODE, TARGET_LANG_CODE)

    if DEBUG_MODE:
        print("DEBUG MODE: Using only first 10 samples.")
        source_sentences = source_sentences[:10]
        target_sentences = target_sentences[:10]

    model, tokenizer = load_llama(quantized=True)

    print(f"Extracting features for {SOURCE_LANG_CODE}...")
    H_source = get_layer_hidden_states(model, tokenizer, source_sentences)

    print(f"Extracting features for {TARGET_LANG_CODE}...")
    H_target = get_layer_hidden_states(model, tokenizer, target_sentences)

    save_path = os.path.join(SAVE_DIR, f"flores_{SOURCE_LANG_CODE}_to_{TARGET_LANG_CODE}.pt")
    torch.save({"source": H_source, "target": H_target}, save_path)
    print(f"Features saved to {save_path}")