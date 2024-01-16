import re
import os
import json
import sys
import pickle
import logging
import argparse
import utils
import pandas as pd
from tqdm import tqdm

from utils import get_children_dir_paths

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def format_jsonl(json_line:list, node_types:list) -> list:
    extract_lines = []

    for line in tqdm(json_line):
        simple_dict = {}
        POSITIVE = 1
        NEGATIVE = 0

        simple_dict["repo"] = line["repo"]
        simple_dict["path"] = line["path"]
        simple_dict["func_name"] = line["func_name"]
        simple_dict["lang"] = line["language"]
        simple_dict["text"] = line["cleaned_code"]
        unique_included_types = line["cleaned_code_subtree_elements_unique"]

        if unique_included_types is None:
            continue

        for node in node_types:
            if node in unique_included_types:
                simple_dict[node] = POSITIVE
            else:
                simple_dict[node] = NEGATIVE
        extract_lines.append(simple_dict)
    
    return extract_lines

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument('--node_types', nargs="*")


    args = parser.parse_args()

    lang_types = get_children_dir_paths(args.input_dir)

    for lang_path in lang_types:
        lang = lang_path.split("/")[-2]
        logging.info(f"=== {lang} ===")

        partition_jsonl_paths = utils.get_jsonl_paths(lang_path)

        for jsonl_path in partition_jsonl_paths:
            with open(jsonl_path, "r") as f:
                json_lines = [json.loads(l) for l in f.readlines()]

            partition_type = jsonl_path.split("/")[-1].split(".")[0]
            logging.info(f"== {partition_type} ==")

            extracted_jsonl = format_jsonl(json_lines, args.node_types)
            logging.info(f"Extracted Data size : {len(extracted_jsonl)}")

            ## Store jsonl files using pandas
            df = pd.DataFrame(extracted_jsonl)

            store_dir = os.path.join(args.output_dir, lang)
            os.makedirs(store_dir, mode=0o777, exist_ok=True)
            store_filename = os.path.join(store_dir, f"{partition_type}.jsonl")

            df.to_json(store_filename, orient="records", force_ascii=False, lines=True)
            logging.info(f"Stored pickle data -> {store_filename}")

    return None


if __name__ == '__main__':
    main()
