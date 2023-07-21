python3 ../run.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 10 \
    --evaluate_step 10 \
    --train_data ../data/astcutting/inputExample/train.pickle \
    --valid_data ../data/astcutting/inputExample/valid.pickle \
    --base_model_save_path ../output/codebert-astcutting

# tmux kill-session -t finetuning_astdiff
