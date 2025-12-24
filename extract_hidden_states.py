import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

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
    all_layer_states = {i: [] for i in range(model.config.num_hidden_layers + 1)}
    print(all_layer_states.__len__())
    # Process in batches to save memory
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
        # all_layer_states.shape: [num_samples, hidden_dim] 
        
    return all_layer_states

source_sentences = [
    "El gato está en la mesa.",  # ES
    "La vie est belle.",         # FR
    "Paka yuko juu ya meza.",    # SW
    "Mero naam John ho."         # NE (Simulated)
]
target_sentences = [
    "The cat is on the table.",
    "Life is beautiful.",
    "The cat is on the table.",
    "My name is John."
]

model, tokenizer = load_llama(quantized=True)
print("Extracting Source States...")
H_source = get_layer_hidden_states(model, tokenizer, source_sentences)

print("Extracting Target States...")
H_target = get_layer_hidden_states(model, tokenizer, target_sentences)

torch.save({"source": H_source, "target": H_target}, "saved_feats_dir/extracted_features.pt")
print("Features saved.")