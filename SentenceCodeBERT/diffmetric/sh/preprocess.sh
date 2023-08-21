CURRENT=$(pwd)

mkdir ../data ../data/CuttingAst ../data/CuttingAst/inputExample
mkdir ../output ../output/CuttingAst 
cd ../data/CuttingAst

gdown 1e188nsXEDsCkIAt7xw8pDXF1M61uvrVz
unzip CuttingAst.zip
rm CuttingAst.zip

cd $CURRENT

lang=python
python3 ../preprocess.py \
    --input_dir ../data/CuttingAst/CuttingAst \
    --output_dir ../data/CuttingAst/inputExample
