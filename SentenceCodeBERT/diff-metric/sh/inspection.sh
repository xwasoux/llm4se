model_path=../output/PrunedAST-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/PrunedAST/inspectInside
mkdir ../output/PrunedAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/PrunedAST-rule_microsoft-codebert-base__max_2023-09-21_21-01-11 \
                ../output/PrunedAST-rule_microsoft-codebert-base__cls_2023-09-21_20-59-33 \
                ../output/PrunedAST-rule_microsoft-codebert-base__mean_2023-09-21_21-02-50 \
    --test_base_dir ../data/PrunedAST/PrunedAST/python/test \
    --lang python \
    --test_data back_seq front_seq \
    --output_path ../output/PrunedAST/inspectOutside

