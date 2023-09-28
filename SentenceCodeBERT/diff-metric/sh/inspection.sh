model_path=../output/PrunedAST-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/PrunedAST/inspectInside
mkdir ../output/PrunedAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/PrunedAST/microsoft-codebert-base_cls_2023-09-27_08-19-44 \
                ../output/PrunedAST/microsoft-codebert-base_max_2023-09-27_08-29-47 \
                ../output/PrunedAST/microsoft-codebert-base_mean_2023-09-27_08-39-50 \
    --test_base_dir ../data/PrunedAST/PrunedAST/python/test \
    --lang python \
    --test_data single_complete \
    --output_path ../output/PrunedAST/inspectOutside

