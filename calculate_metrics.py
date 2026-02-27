import argparse
import json
import os
import torch
import sacrebleu
from comet import download_model, load_from_checkpoint

COMET_MODEL_DEFAULT = "Unbabel/wmt22-comet-da"
AFRICOMET_MODEL_ID = "masakhane/africomet-stl-1.1"

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

    # Basic cleanup
    hypotheses = [h.split("(")[-1] for h in hypotheses]
    hypotheses = [h.split(":")[-1] for h in hypotheses]
    hypotheses = [h.split('\n')[0] for h in hypotheses]
            
    return sources, hypotheses, references

def calculate_spbleu(hypotheses, references):
    """
    Calculates BLEU score.
    For English targets, tokenizer='13a' is standard and comparable to spBLEU reports.
    If evaluating non-English, consider using tokenizer='spm' if you have the model,
    or tokenizer='flores101' if supported.
    """
    # SacreBLEU expects list of lists for references
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize='13a')
    return bleu.score

def calculate_chrf_pp(hypotheses, references):
    """
    Calculates chrF++ score.
    chrF++ is simply chrF with word_order=2.
    """
    # chrF score with word_order=2 is equivalent to chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    return chrf.score

def calculate_comet_variant(sources, hypotheses, references, model_id, gpus=1):
    """
    Calculates COMET scores using the specified model_id.
    """
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]
    
    print(f"\nLoading COMET model ({model_id})...")
    try:
        model_path = download_model(model_id)
        model = load_from_checkpoint(model_path)
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return None
    
    print(f"Computing scores with {model_id}...")
    model_output = model.predict(data, batch_size=8, gpus=gpus)
    
    return model_output.system_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate Translation Output with Multiple Metrics")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .jsonl output file")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU usage for COMET/AfriCOMET")
    
    # Metric selection arguments
    parser.add_argument("--metrics", nargs='+', 
                        choices=['spbleu', 'chrf', 'comet', 'africomet', 'all'], 
                        default=['all'],
                        help="List of metrics to calculate. Options: spbleu, chrf, comet, africomet, all")

    args = parser.parse_args()
    
    # Determine which metrics to run
    metrics_to_run = set(args.metrics)
    if 'all' in metrics_to_run:
        metrics_to_run = {'spbleu', 'chrf', 'comet', 'africomet'}

    sources, hypotheses, references = load_data(args.input_file)
    print(f"Loaded {len(sources)} sentences.")
    
    results = {}
    
    # spBLEU 
    if 'spbleu' in metrics_to_run:
        score = calculate_spbleu(hypotheses, references)
        print(f"--- spBLEU Score: {score:.2f} ---")
        results['spBLEU'] = score

    # chrF++ 
    if 'chrf' in metrics_to_run:
        score = calculate_chrf_pp(hypotheses, references)
        print(f"--- chrF++ Score: {score:.2f} ---")
        results['chrF++'] = score

    # Setup GPU for Neural Metrics
    gpus = 0 if args.no_cuda or not torch.cuda.is_available() else 1

    # COMET 
    if 'comet' in metrics_to_run:
        score = calculate_comet_variant(sources, hypotheses, references, COMET_MODEL_DEFAULT, gpus)
        if score is not None:
            print(f"--- COMET Score:  {score:.4f} ---")
            results['COMET'] = score

    # AfriCOMET 
    if 'africomet' in metrics_to_run:
        score = calculate_comet_variant(sources, hypotheses, references, AFRICOMET_MODEL_ID, gpus)
        if score is not None:
            print(f"--- AfriCOMET Score: {score:.4f} ---")
            results['AfriCOMET'] = score

    # Save Results 
    summary_folder = os.path.dirname(args.input_file)
    summary_path = os.path.join(summary_folder, "results.txt")
    
    with open(summary_path, "a") as f:
        f.write('\n')
        f.write(f"File: {args.input_file}\n")
        for metric, val in results.items():
            if isinstance(val, float):
                f.write(f"{metric}: {val:.4f}\n")
            else:
                f.write(f"{metric}: {val}\n")
        f.write('-' * 20 + '\n')

    print(f"\nScores saved to {summary_path}")

if __name__ == "__main__":
    main()