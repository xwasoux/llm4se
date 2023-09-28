model_path=../output/PrunedAST-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/PrunedAST/inspectInside
mkdir ../output/PrunedAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/PrunedAST/microsoft-codebert-base_cls_2023-09-28_16-42-51 \
                ../output/PrunedAST/microsoft-codebert-base_max_2023-09-28_16-52-46 \
                ../output/PrunedAST/microsoft-codebert-base_mean_2023-09-28_17-02-35 \
    --test_base_dir ../data/PrunedAST/PrunedAST/python/test \
    --lang python \
    --test_data single_complete \
    --output_path ../output/PrunedAST/inspectOutside

