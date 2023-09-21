model_path=../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/CuttingAST/inspectInside
mkdir ../output/CuttingAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/CuttingAST-rule_microsoft-codebert-base__max_2023-09-21_21-01-11 \
                ../output/CuttingAST-rule_microsoft-codebert-base__cls_2023-09-21_20-59-33 \
                ../output/CuttingAST-rule_microsoft-codebert-base__mean_2023-09-21_21-02-50 \
    --test_base_dir ../data/CuttingAST/CuttingAST/python/test \
    --lang python \
    --test_data back_seq front_seq \
    --output_path ../output/CuttingAST/inspectOutside

