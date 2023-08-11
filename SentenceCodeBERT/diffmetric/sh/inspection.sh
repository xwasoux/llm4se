model_path=../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08

mkdir ../output/CuttingAst/inspectInside
mkdir ../output/CuttingAst/inspectOutside

python3 ../inspection.py \
    --model_path ../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08 \
    --test_base_dir ../data/CuttingAst/CuttingAst/python/test \
    --lang python \
    --test_data front_seq \
    --output_dir inspectInside 

python3 ../inspection.py \
    --model_path ../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08 \
    --test_base_dir ../data/CuttingAst/CuttingAst/python/test \
    --lang python \
    --test_data rule_point \
    --output_dir inspectOutside 

python3 ../inspection.py \
    --model_path ../output/cuttingAst-random_microsoft-codebert-base_2023-08-06_08-24-08 \
    --test_base_dir ../data/CuttingAst/CuttingAst/python/test \
    --lang python \
    --test_data back_seq \
    --output_dir inspectOutside 
