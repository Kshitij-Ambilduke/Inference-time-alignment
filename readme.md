# Cross-lingual intervention for low-resource Machine Translation
### Kshitij Ambilduke & Shubha Sanket Samantaray 

This repository contains the code for the project for the course on Deep Learning at MVA, 2025-2026. The project is based around the paper *Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention*[(Link)](https://arxiv.org/abs/2410.12462).

## Pre-requisites

Install all the necessary packages either using `conda` or in a virtual environment.
```bash
pip install -r requirements.txt 
``` 

## Running the code
The complete pipeline to run the experiments can be found in the `pipeline.sh` file. Shortly, the training process involves 3 steps in particular:
1. Extract hidden features:  To train the alignement matrices, we need to extract the representations of the last token in the sequence. Run `extract_hidden_features.py` for this.
```bash
python extract_hidden_features.py \
    --source_lang npi_Deva \
    --target_lang eng_Latn \
    --save_dir results/nepali_to_en \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 64 \
    --samples 500
    # --debug # for debugging
    # --no_quant # for not applying quantization
```

2. Training alignment matrices: Depending on whether to train the vanilla INCLINE method (`train_incline.py`) or the Improved INCLINE method (`train_improved_incline.py`), run the file accordingly.
3. Inference: After the matrices are trained, we need to create a forward hook to intervene and insert the new cross-lingual embeddings. Use `inference_incline_batch.py` for this.
```bash
python train_incline.py \
    --input_file "flores_npi_Deva_to_eng_Latn.pt" \
    --output_file "flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt" \
    --lam 0.01 # L2 norm weight
```

```bash
python inference_incline_batch.py \
    --source-lang "npi_Deva" \
    --target-lang "fra_Latn" \
    --matrix-path "results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt" \
    --alpha 0.4 \
    --output-folder "results/nepali_to_en" \
    --model-id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch-size 4 \
    # --debug  # set this for debugging quickly
```

Besides this, we also provide code for the baseline generation `inference_baseline.py` which contains code to translate from source to target without any intervention or training, using just the base model. We also provide the code for topline model training `train_lora.py` and inference `inference_lora.py`.

Finally, all the evaluation code is in `calculate_metrics.py`.
