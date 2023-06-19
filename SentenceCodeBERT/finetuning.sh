python3 semanticEastSim.py \
    --model_name_or_path microsoft/codebert-base \
    --train_batch_size 2 \
    --epochs_num 100 \
    --evaluate_step 100

tmux kill-session -t sentencecodebert