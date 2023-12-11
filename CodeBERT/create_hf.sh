dir_name=$1

mkdir $dir_name
cd $dir_name
main_dir=$(pwd)

mkdir data data/inputs 
mkdir output
mkdir sh

cd sh
touch preprocess.sh finetuning.sh inspect.sh

cd $main_dir
touch run.py preprocess.py inspect.py utils.py
