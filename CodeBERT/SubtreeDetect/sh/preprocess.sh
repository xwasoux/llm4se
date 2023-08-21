CURRENT=$(pwd)

cd ../data

gdown 1pj_fzfcl77230ydB-qUQw2oxcEBUBMw8
unzip SubtreeElement.zip
rm  SubtreeElement.zip

cd $CURRENT

lang=python
python3 ../preprocess.py \
    --input_dir ../data/SubtreeElement \
    --output_dir ../data/inputs \
    --node_types if_statement elif_clause else_clause for_statement while_statement expression_statement return_statement break_statement with_statement \
