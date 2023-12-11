python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/PrunedAST/inputExample \
    --language python \
    --train_datas single_selected sequence_forward sequence_backward single_complete \
    --base_model_save_path ../output/PrunedAST \
    --pooling_mode_cls \
    --upper_data_size 300

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/PrunedAST/inputExample \
    --language python \
    --train_datas single_selected sequence_forward sequence_backward single_complete \
    --base_model_save_path ../output/PrunedAST \
    --pooling_mode_max \
    --upper_data_size 300

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/PrunedAST/inputExample \
    --language python \
    --train_datas single_selected sequence_forward sequence_backward single_complete \
    --base_model_save_path ../output/PrunedAST \
    --pooling_mode_mean \
    --upper_data_size 300