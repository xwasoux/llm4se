python3 ../semanticCodeSim.py \
    --model_name_or_path microsoft/codebert-base \
    --do_train \
    --train_batch_size 2 \
    --epochs_num 100 \
    --evaluate_step 100 \
    --datasets /workspace/SentenceCodeBERT/data/semantic_east_sim/semantic_east_sim_dataset.pickle \
    --base_model_save_path output/codebert_east_diff

tmux kill-session -t east_metric_learning