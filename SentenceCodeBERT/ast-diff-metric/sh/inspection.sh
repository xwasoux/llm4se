model_path=../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/CuttingAST/inspectInside
mkdir ../output/CuttingAST/inspectOutside

python3 ../inspection.py \
    --model_path ../output/CuttingAST/microsoft-codebert-base_cls_2023-09-24_09-21-32 \
                ../output/CuttingAST/microsoft-codebert-base_max_2023-09-24_09-31-29 \
                ../output/CuttingAST/microsoft-codebert-base_mean_2023-09-24_15-20-53 \
                ../output/CuttingAST/microsoft-codebert-base_cls-max_2023-09-24_15-40-43 \
                ../output/CuttingAST/microsoft-codebert-base_cls-mean_2023-09-24_15-30-51 \
                ../output/CuttingAST/microsoft-codebert-base_max-mean_2023-09-24_15-50-40 \
                ../output/CuttingAST/microsoft-codebert-base_cls-max-mean_2023-09-24_16-00-42 \
    --test_base_dir ../data/CuttingAST/CuttingAST/python/test \
    --lang python \
    --test_data all_point \
    --output_path ../output/CuttingAST/inspectOutside

