import argparse
import json
import os
import torch
import sacrebleu
from comet import download_model, load_from_checkpoint

def load_data(file_path):
    """
    Reads the .jsonl file and extracts sources, hypotheses (generated), and references.
    """
    sources = []
    hypotheses = []
    references = []
    
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sources.append(data["source"])
            hypotheses.append(data["generated"])
            references.append(data["reference"])
            
    return sources, hypotheses, references

def calculate_sacrebleu(hypotheses, references):
    """
    Calculates the BLEU score using sacrebleu.
    """
    # SacreBLEU expects references to be a list of lists (for multi-ref support)
    # Since we have 1 reference per sentence, we wrap the list of references in another list.
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score

def calculate_comet(sources, hypotheses, references, gpus=1):
    """
    Calculates the COMET score using the wmt22-comet-da model.
    """
    # COMET requires data in a specific dictionary format
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    
    print("\nLoading COMET model (Unbabel/wmt22-comet-da)...")
    # This downloads the model to a local cache if not present
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    # Calculate scores
    print("Computing COMET scores...")
    model_output = model.predict(data, batch_size=8, gpus=gpus)
    
    # model_output.system_score is the average score for the whole dataset
    return model_output.system_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate Translation Output with SacreBLEU and COMET")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .jsonl output file")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU usage for COMET")
    
    args = parser.parse_args()
    
    sources, hypotheses, references = load_data(args.input_file)
    print(f"Loaded {len(sources)} sentences.")

    bleu_score = calculate_sacrebleu(hypotheses, references)
    print(f"\n--- SacreBLEU Score: {bleu_score:.2f} ---")

    gpus = 0 if args.no_cuda or not torch.cuda.is_available() else 1
    
    comet_score = calculate_comet(sources, hypotheses, references, gpus=gpus)
    print(f"--- COMET Score:     {comet_score:.4f} ---")

    summary_path = args.input_file.replace(".jsonl", "_scores.txt")
    with open(summary_path, "w") as f:
        f.write(f"File: {args.input_file}\n")
        f.write(f"SacreBLEU: {bleu_score:.2f}\n")
        f.write(f"COMET: {comet_score:.4f}\n")
    print(f"\nScores saved to {summary_path}")

if __name__ == "__main__":
    main()