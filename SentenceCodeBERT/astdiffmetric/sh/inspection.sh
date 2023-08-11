mkdir ../output/astcutting/inspectInside
mkdir ../output/astcutting/inspectOutside

python3 ../inspection.py \
    --model_path ../output/codebert-astcutting_microsoft-codebert-base_2023-07-21_03-35-10 \
    --test_data ../data/astcutting/astcutting/train \
    --output_dir ../output/astcutting/inspectInside \
    --upper_char_size 850

python3 ../inspection.py \
    --model_path ../output/codebert-astcutting_microsoft-codebert-base_2023-07-21_03-35-10 \
    --test_data ../data/astcutting/astcutting/test \
    --output_dir ../output/astcutting/inspectOutside \
    --upper_char_size 850