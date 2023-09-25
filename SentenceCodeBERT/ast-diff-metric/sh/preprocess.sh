CURRENT=$(pwd)

mkdir ../data ../data/CuttingAST ../data/CuttingAST/inputExample
mkdir ../output ../output/CuttingAST 
cd ../data/CuttingAST

gdown 1Qzfh7exLYi5VrICS0syPh9h32Siab76q # 500 programs
unzip CuttingAST.zip
rm CuttingAST.zip

cd $CURRENT

lang=python
python3 ../preprocess.py \
    --input_dir ../data/CuttingAST/CuttingAST \
    --output_dir ../data/CuttingAST/inputExample
