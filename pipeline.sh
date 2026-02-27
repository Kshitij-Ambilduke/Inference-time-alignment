# extracting features from each layers and saving them
python extract_hidden_features.py \
    --source_lang npi_Deva \
    --target_lang eng_Latn \
    --save_dir results/nepali_to_en \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 64 \
    --samples 500
    # --debug # for debugging
    # --no_quant # for not applying quantization

# train INCLINE with regularized ridge regression
python train_incline.py \
    --input_file "/home/shishirk/adityasr/kshitij/results/nepali_to_en/flores_npi_Deva_to_eng_Latn.pt" \
    --output_file "/home/shishirk/adityasr/kshitij/results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt" \
    --lam 0.01

# generate outputs from INCLINE model
python inference_incline.py \
    --source-lang "npi_Deva" \
    --target-lang "fra_Latn" \
    --matrix-path "results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt" \
    --alpha 0.4 \
    --output-folder "results/nepali_to_en" \
    --model-id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --debug

# batch generate try INCLINE model
python inference_incline_batch.py \
    --source-lang "npi_Deva" \
    --target-lang "fra_Latn" \
    --matrix-path "results/nepali_to_en/flores_npi_Deva_to_eng_Latn_INCLINE_MATRICES.pt" \
    --alpha 0.4 \
    --output-folder "results/nepali_to_en" \
    --model-id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --debug \
    --batch-size 4

# generate outputs from BASELINE model
python inference_baseline.py \
  --source_lang npi_Deva \
  --output_dir results/nepali_to_en \
  --batch_size 256 \
  --debug

# evaluation for INCLINE model
python calculate_metrics.py \
    --input_file "results/nepali_to_en/incline_npi_Deva_alpha0.4.jsonl"

# evaluation for BASELINE model
python calculate_metrics.py \
    --input_file "results/nepali_to_en/baseline_npi_Deva_to_eng_Latn.jsonl"