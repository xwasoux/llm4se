model_path=../output/PrunedAST-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/PrunedAST/inspectInside
mkdir ../output/PrunedAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/PrunedAST/microsoft-codebert-base_cls_2023-09-24_09-21-32 \
                ../output/PrunedAST/microsoft-codebert-base_max_2023-09-24_09-31-29 \
                ../output/PrunedAST/microsoft-codebert-base_mean_2023-09-24_15-20-53 \
                ../output/PrunedAST/microsoft-codebert-base_cls-max_2023-09-24_15-40-43 \
                ../output/PrunedAST/microsoft-codebert-base_cls-mean_2023-09-24_15-30-51 \
                ../output/PrunedAST/microsoft-codebert-base_max-mean_2023-09-24_15-50-40 \
                ../output/PrunedAST/microsoft-codebert-base_cls-max-mean_2023-09-24_16-00-42 \
    --test_base_dir ../data/PrunedAST/PrunedAST/python/test \
    --lang python \
    --test_data point_all \
    --output_path ../output/PrunedAST/inspectOutside

