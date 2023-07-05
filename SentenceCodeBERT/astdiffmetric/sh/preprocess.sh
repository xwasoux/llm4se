mkdir ../data ../data/semantic_east_sim
cd ../data/semantic_east_sim
gdown https://drive.google.com/uc?id=1n6OyDsNifUY0h9h8yNpWzvdkseqSly-9
unzip semantic_east_sim_data.zip
rm  semantic_east_sim_data.zip
cd ../../semantic_east_sim
python ast_diff_preprocess.py
cd ..