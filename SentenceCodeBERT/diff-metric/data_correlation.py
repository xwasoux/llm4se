import os
import csv
import numpy as np
import pandas as pd
import tree_sitter
from tree_sitter import Node, Language, Parser

from astars import AParser, ATraverser

def calculate_correlation(series1, series2):
    if len(series1) != len(series2):
        raise ValueError("Both series must have the same length.")
    correlation_coefficient = np.corrcoef(series1, series2)[0, 1]
    
    return correlation_coefficient

def code_parse(edited_code):
    parser = AParser()
    traverser = ATraverser()
    tree = parser.parse(text=edited_code, lang="python")
    node = traverser.preorderTraverse(tree)
    for type in node.preNodeTypes:
        if type == "ERROR":
            return False
    return True

def extract_data_from_csv(csv_file, edited_code_colum, inspect_score_colum):
    score_list = []
    
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        
        for row in reader:
            if code_parse(row[edited_code_colum]):
                score_list.append(float(row[inspect_score_colum]))
                
    return score_list  

back_csv_path = "/workspace/SentenceCodeBERT/diff-metric/output/PrunedAST/inspectOutside/sequence_backward_fs/2023-09-28_16-05-36_sequence-backward_cls-max-mean.csv"
complete_csv_path = "/workspace/SentenceCodeBERT/diff-metric/output/PrunedAST/inspectOutside/single_complete_fs/2023-09-28_16-09-22_single-complete_cls-max-mean.csv"
cls_score = np.array(extract_data_from_csv(complete_csv_path, 12, 51))
max_score = np.array(extract_data_from_csv(complete_csv_path, 12, 52))
mean_score = np.array(extract_data_from_csv(complete_csv_path, 12, 53))
diff_char_size = np.array(extract_data_from_csv(complete_csv_path, 12, 40))

cls_correlation_coefficient = calculate_correlation(cls_score, diff_char_size)
max_correlation_coefficient = calculate_correlation(max_score, diff_char_size)
mean_correlation_coefficient = calculate_correlation(mean_score, diff_char_size)

print("cls : ", cls_correlation_coefficient)
print("max : ", max_correlation_coefficient)
print("mean : ", mean_correlation_coefficient)