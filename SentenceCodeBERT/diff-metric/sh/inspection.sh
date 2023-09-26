model_path=../output/PrunedAST-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/PrunedAST/inspectInside
mkdir ../output/PrunedAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/PrunedAST/microsoft-codebert-base_cls_2023-09-23_16-28-01 \
                ../output/PrunedAST/microsoft-codebert-base_max_2023-09-23_16-36-43 \
                ../output/PrunedAST/microsoft-codebert-base_mean_2023-09-23_16-45-29 \
                ../output/PrunedAST/microsoft-codebert-base_cls-max_2023-09-23_17-02-52 \
                ../output/PrunedAST/microsoft-codebert-base_cls-mean_2023-09-23_16-54-10 \
                ../output/PrunedAST/microsoft-codebert-base_max-mean_2023-09-23_17-11-35 \
                ../output/PrunedAST/microsoft-codebert-base_cls-max-mean_2023-09-23_17-20-20 \
    --test_base_dir ../data/PrunedAST/PrunedAST/python/test \
    --lang python \
    --test_data all_point \
    --output_path ../output/PrunedAST/inspectOutside

