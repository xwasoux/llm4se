python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_datas rule_point back_seq \
    --base_model_save_path ../output/CuttingAST \
    --pooling_mode_cls \
    --upper_data_size 100

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_datas rule_point back_seq \
    --base_model_save_path ../output/CuttingAST \
    --pooling_mode_max \
    --upper_data_size 100

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_datas rule_point back_seq \
    --base_model_save_path ../output/CuttingAST \
    --pooling_mode_mean \
    --upper_data_size 100
