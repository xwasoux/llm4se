import os
import json
import sys
import glob
import pickle
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from sentence_transformers import InputExample

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def get_children_dir_paths(path_name:str) -> list:
    condition = f'{path_name}/*/'
    return glob.glob(condition)

def get_jsonl_paths(path_name:str) -> list:
    condition = f'{path_name}/*.jsonl'
    return glob.glob(condition)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    lang_types = get_children_dir_paths(args.input_dir)

    for lang_path in lang_types:
        lang = lang_path.split("/")[-2]
        logging.info(f"=== {lang} ===")

        all_partition_paths = get_children_dir_paths(lang_path)

        for partition_path in all_partition_paths:
            partition_type = partition_path.split("/")[-2]
            logging.info(f"== {partition_type} ==")

            logging.info(partition_path)
            
            pruning_type_paths = get_children_dir_paths(partition_path)
            logging.info(pruning_type_paths)

            for path in pruning_type_paths:
                pruning_type = path.split("/")[-2]

                all_jsonl_paths = get_jsonl_paths(path)

                all_input_examples = []
                for jsonl_path in tqdm(all_jsonl_paths):
                    with open(jsonl_path) as f:
                        jsonl_data = [json.loads(l) for l in f.readlines()]

                    for line in jsonl_data:
                        all_input_examples.append(InputExample(guid=f"index", 
                                                                texts=[line["originalCode"], line["editedCode"]], 
                                                                label=line["cosSimChar"]))

                logging.info(f"InputExample Data : {len(all_input_examples)}")

                store_dir = os.path.join(args.output_dir, lang, partition_type)
                os.makedirs(store_dir, mode=0o777, exist_ok=True)
                store_file_name = os.path.join(store_dir, f"{pruning_type}.pickle")
                with open(store_file_name, "wb") as p:
                    pickle.dump(all_input_examples, p)
                logging.info(f"Stored pickle data -> {store_file_name}")

    return None


if __name__ == '__main__':
    main()

