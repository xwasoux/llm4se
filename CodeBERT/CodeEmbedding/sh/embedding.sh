## CodeBERT embedding
python ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --torch_transformer \
    --csv_data_path ../data/input.csv \
    --dimention 2 \
    --outputdir ../output