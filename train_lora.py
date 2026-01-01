import torch
import os
import argparse
import random
from typing import Dict, List, Any

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq  # Handles padding labels to -100 automatically
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset

# --- Constants ---
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

def load_flores_pair_for_training(src_lang, tgt_lang="eng_Latn", split="dev", samples=500):
    print(f"Loading FLORES dataset: {src_lang} -> {tgt_lang} ({split})")
    
    ds_src = load_dataset("facebook/flores", src_lang, split=split, trust_remote_code=True)
    ds_tgt = load_dataset("facebook/flores", tgt_lang, split=split, trust_remote_code=True)
    
    combined = list(zip(ds_src['sentence'], ds_tgt['sentence']))
    random.shuffle(combined)
    
    if samples < len(combined):
        combined = combined[:samples]
        
    print(f"Loaded {len(combined)} pairs for training.")

    # Return as a list of dicts
    return [{"src": src, "tgt": tgt} for src, tgt in combined]

def preprocess_and_mask(examples, tokenizer, src_lang_code):
    """
    1. Formats the text.
    2. Tokenizes the full sequence.
    3. Creates 'labels' where the Instruction part is masked with -100.
    """
    src_lang_name = LANG_ID_TO_NAME.get(src_lang_code, src_lang_code)
    
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for src, tgt in zip(examples['src'], examples['tgt']):
        # 1. Construct the Instruction (Prompt) and the Full Text
        # Note: We add a space after 'English:' to ensure clean tokenization boundaries
        instruction = f"Translate the following {src_lang_name} source sentence to English\nSource: {src}\nEnglish: "
        full_text = instruction + tgt + tokenizer.eos_token
        
        # 2. Tokenize (No padding yet, we pad in the collator)
        # We use standard tokenization. 
        # Ideally, we want the model to predict the target *continuing* from the prompt.
        
        tokenized_full = tokenizer(full_text, truncation=True, max_length=256, add_special_tokens=True)
        tokenized_instr = tokenizer(instruction, truncation=True, max_length=256, add_special_tokens=True)
        
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        
        # 3. Create Labels
        # Start with a copy of input_ids
        labels = input_ids[:]
        
        # Calculate length of the instruction tokens
        # We subtract 1 sometimes if the tokenizer merges the start token, 
        # but usually len(tokenized_instr) is the safe split point.
        instr_len = len(tokenized_instr["input_ids"])
        
        # Mask the instruction part
        # If the instruction is longer than the full text (truncation edge case), mask everything
        mask_len = min(instr_len, len(labels))
        
        # Set indices [0 ... mask_len] to -100
        # However, we usually want the model to predict the first token of the answer *given* the last token of the prompt.
        # So we often mask up to mask_len-1. But strictly masking the whole instruction is safer.
        for i in range(mask_len):
            labels[i] = -100
            
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
        
    return model_inputs

def train_lora_baseline(args):
    # 1. Load Data
    raw_data = load_flores_pair_for_training(
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        samples=args.samples
    )
    dataset = Dataset.from_list(raw_data)

    # 2. Configure Quantization
    if args.no_quant:
        print("Loading model in bfloat16 (No Quantization)...")
        bnb_config = None
        torch_dtype = torch.bfloat16
    else:
        print("Loading model in 4-bit NF4 (Quantized)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        torch_dtype = None

    # 3. Load Model
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    if not args.no_quant:
        model = prepare_model_for_kbit_training(model)

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        r=16,       
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Tokenizer & Pre-processing
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # For Training, we use Right Padding because the DataCollator expects it
    tokenizer.padding_side = "right" 

    print("Tokenizing and Masking dataset...")
    # We use batched=True to process lists of examples
    processed_dataset = dataset.map(
        lambda x: preprocess_and_mask(x, tokenizer, args.source_lang),
        batched=True,
        remove_columns=dataset.column_names # Remove raw text columns, keep only tensors
    )

    # 6. Data Collator
    # DataCollatorForSeq2Seq is excellent for CausalLM training too because:
    # It dynamically pads 'input_ids' to the max length in the batch.
    # It dynamically pads 'labels' with -100 (ignore_index) to match the sequence length.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model, # It grabs pad_token_id from here
        padding="longest"
    )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=args.max_steps, 
        learning_rate=2e-4,
        fp16=True, 
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="no",       
        report_to="none",
        remove_unused_columns=False # Important when using custom columns like 'labels' with LoRA
    )

    # 8. Trainer (Standard HF Trainer)
    trainer = Trainer(
        model=model,
        train_dataset=processed_dataset,
        args=training_args,
        data_collator=data_collator
    )

    # 9. Train
    print("Starting LoRA training...")
    trainer.train()

    # 10. Save
    print(f"Saving adapter to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA baseline (Standard Trainer).")
    
    parser.add_argument("--source_lang", type=str, required=True, help="Source language code (e.g., npi_Deva)")
    parser.add_argument("--target_lang", type=str, default="eng_Latn", help="Target language code")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="results/lora_baseline")
    parser.add_argument("--samples", type=int, default=500, help="Number of training pairs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization.")

    args = parser.parse_args()
    
    train_lora_baseline(args)