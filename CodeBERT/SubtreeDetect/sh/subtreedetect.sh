python3 ../subtreedetect.py \
    --target_node_types if_statement elif_clause else_clause for_statement while_statement expression_statement return_statement break_statement with_statement \
    --language python \
    --input_base_dir ../data/SubtreeDetection \
    --pretrained_model_base_dir ../output/model \
    --output_base_dir ../output/SubtreeDetection

