## CodeBERT embedding
python ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --torch_transformer \
    --data_dir ../data/input.csv \
    --dim 2 \
    --output_dir ../output