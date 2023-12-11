CURRENT=$(pwd)

mkdir ../data ../data/PrunedAST ../data/PrunedAST/inputExample
mkdir ../output ../output/PrunedAST
cd ../data/PrunedAST

# gdown 1Qzfh7exLYi5VrICS0syPh9h32Siab76q # 500 programs
# unzip PrunedAST.zip
# rm PrunedAST.zip
gdown 1hwYgOMeM22iah_LuPzlxqRzbUhtQNdF8 # 500 programs
unzip PrunedAST.zip
rm PrunedAST.zip

cd $CURRENT

lang=python
python3 ../preprocess.py \
    --input_dir ../data/PrunedAST/PrunedAST \
    --output_dir ../data/PrunedAST/inputExample
