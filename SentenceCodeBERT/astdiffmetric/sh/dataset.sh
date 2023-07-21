mkdir ../data ../data/astcutting ../data/astcutting/inputExample
mkdir ../output ../output/astcutting ../output/astcutting/inputExample
cd ../data/astcutting

gdown 13dcE8Pj44JHG0cPHVK8cZjw-lN0DEGNI
unzip astcutting.zip
rm astcutting.zip
cd ../..

python3 preprocess.py \
    --train_data ./data/astcutting/astcutting/train \
    --valid_data ./data/astcutting/astcutting/valid \
    --test_data ./data/astcutting/astcutting/test \
    --upper_char_size 800 \
    --output_dir ./data/astcutting/inputExample 

cd sh/