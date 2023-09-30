# back_csv_path = /workspace/SentenceCodeBERT/diff-metric/output/PrunedAST/inspectOutside/sequence_backward_fs/2023-09-28_16-05-36_sequence-backward_cls-max-mean.csv
# complete_csv_path = /workspace/SentenceCodeBERT/diff-metric/output/PrunedAST/inspectOutside/single_complete_fs/2023-09-28_16-09-22_single-complete_cls-max-mean.csv

python3 ../data_correlation.py \
    --csv_path ../output/PrunedAST/inspectOutside/sequence_backward_fs/2023-09-28_16-05-36_sequence-backward_cls-max-mean.csv \
    --output_filtered_csv