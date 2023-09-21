python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_data rule_point \
    --base_model_save_path ../output/CuttingAST-rule \
    --pooling_mode_cls

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_data rule_point \
    --base_model_save_path ../output/CuttingAST-rule \
    --pooling_mode_max

python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAST/inputExample \
    --language python \
    --train_data rule_point \
    --base_model_save_path ../output/CuttingAST-rule \
    --pooling_mode_mean
