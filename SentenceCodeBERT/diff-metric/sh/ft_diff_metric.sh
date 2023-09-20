python3 ../semanticCodeSim.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 100 \
    --evaluate_step 100 \
    --datasets /workspace/SentenceCodeBERT/data/semantic_diff_sim/dezero_diff_dataset.pickle \
    --base_model_save_path ../output/codebert-diff

tmux kill-session -t diff_metric_learning