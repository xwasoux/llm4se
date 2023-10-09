import os
import csv
import ast
import numpy as np
import pandas as pd
import argparse
import tree_sitter
from tree_sitter import Node, Language, Parser

from astars import AParser, ATraverser

def calculate_correlation(series1:np.ndarray, series2:np.ndarray) -> float:
    if len(series1) != len(series2):
        raise ValueError("Both series must have the same length.")
    correlation_coefficient = np.corrcoef(series1, series2)[0, 1]
    
    return correlation_coefficient

def parse_judge(edited_code:str) -> bool:
    try:
        res = ast.parse(edited_code)
        return True
    except SyntaxError:
        return False
    

def extract_data_from_csv(csv_file:str, edited_code_label:str, inspect_score_label:str) -> list:
    score_list = []
    
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        
        try:
            edited_code_colum = header.index(edited_code_label)
            #print(edited_code_colum)
            inspect_score_colum = header.index(inspect_score_label)
        except ValueError:
            raise ValueError("Column labels not found in the CSV header.")
        
        for row in reader:
            if parse_judge(row[edited_code_colum]):
                score_list.append(float(row[inspect_score_colum]))
                                  
    return score_list  

def extract_code_from_csv(csv_file:str) -> list:
    cleaned_code_list = []
    edited_code_list = []
    
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        
        cleaned_code_label = "cleaned_code"
        edited_code_label = "edited_code"
        
        try:
            cleaned_code_colum = header.index(cleaned_code_label)
            edited_code_colum = header.index(edited_code_label)
        except ValueError:
            raise ValueError("Column labels not found in the CSV header.")
        
        for row in reader:
            cleaned_code_list.append(row[cleaned_code_colum])
            edited_code_list.append(row[edited_code_colum])
            
    return cleaned_code_list, edited_code_list
                
def create_score_list(csv_path:str) -> np.array:
    cls_score = np.array(extract_data_from_csv(csv_path, "edited_code", "inspect_cls"))
    max_score = np.array(extract_data_from_csv(csv_path, "edited_code", "inspect_max"))
    mean_score = np.array(extract_data_from_csv(csv_path, "edited_code", "inspect_mean"))
    
    diff_char_size = np.array(extract_data_from_csv(csv_path, "edited_code", "cleaned_code_diff_char_size"))
    
    return cls_score, max_score, mean_score, diff_char_size

def calculate_and_print_correlations(csv_path:str) -> None:
    result = create_score_list(csv_path)
    cls_score, max_score, mean_score, diff_char_size = result
    cls_correlation_coefficient = calculate_correlation(cls_score, diff_char_size)
    max_correlation_coefficient = calculate_correlation(max_score, diff_char_size)
    mean_correlation_coefficient = calculate_correlation(mean_score, diff_char_size)

    print("cls : ", cls_correlation_coefficient)
    print("max : ", max_correlation_coefficient)
    print("mean : ", mean_correlation_coefficient)

def export_filtered_csv(csv_path: str) -> None:
    data_result = create_score_list(csv_path)
    code_result = extract_code_from_csv(csv_path)
    
    cls_score, max_score, mean_score, diff_char_size = data_result
    cleaned_code, edited_code = code_result
    
    file_name = os.path.basename(csv_path)
    
    parent_directory = os.path.dirname(csv_path)
    new_directory = os.path.join(parent_directory, "filter_" + file_name)
    os.makedirs(new_directory, exist_ok=True)
    new_csv_file_path = os.path.join(new_directory, "filter_" + file_name)
    
    with open(new_csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        header = ["cleaned_code_diff_char_size","cleaned_code", "edited_code", "inspect_cls", "inspect_max", "inspect_mean"]
        writer.writerow(header)

        for i in range(len(cls_score)):
            row = [diff_char_size[i], cleaned_code[i], edited_code[i], cls_score[i], max_score[i], mean_score[i]]
            writer.writerow(row)
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--output_correlation', action="store_true")
    parser.add_argument('--output_filtered_csv', action="store_true")

    args = parser.parse_args()
    
    if args.output_correlation:
        calculate_and_print_correlations(args.csv_path)
    else:
        pass
    
    if args.output_filtered_csv:
        export_filtered_csv(args.csv_path)
    else:
        pass
    
    
if __name__ == '__main__':
    main()