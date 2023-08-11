python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --input_base_dir ../data/CuttingAst/inputExample \
    --language python \
    --train_data back_seq front_seq \
    --base_model_save_path ../output/cuttingAst-random
