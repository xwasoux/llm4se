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


def get_dir_paths(path_name:str) -> list:
    condition = f'{path_name}/*/'
    return glob.glob(condition)

def get_jsonl_paths(path_name:str) -> list:
    condition = f'{path_name}/*.jsonl'
    return glob.glob(condition)

def mk_dir(path_str:str) -> None:
    os.makedirs(path_str, mode=0o777, exist_ok=True)
    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    lang_types = get_dir_paths(args.input_dir)

    for lang_path in lang_types:
        lang = lang_path.split("/")[-2]
        logging.info(f"=== {lang} ===")

        each_types_paths = get_dir_paths(lang_path)

        for purpose_path in each_types_paths:
            purpose_type = purpose_path.split("/")[-2]
            logging.info(f"== {purpose_type} ==")

            logging.info(purpose_path)
            
            deletion_type_paths = get_dir_paths(purpose_path)
            logging.info(deletion_type_paths)

            for delete_type_path in deletion_type_paths:
                delete_type = delete_type_path.split("/")[-2]

                all_data_path = get_jsonl_paths(delete_type_path)

                all_input_examples = []

                for data_path in tqdm(all_data_path):
                    with open(data_path) as f:
                        jsonl_data = [json.loads(l) for l in f.readlines()]

                    for line in jsonl_data:
                        all_input_examples.append(InputExample(guid=f"index", 
                                                            texts=[line["originalCode"], line["editedCode"]], 
                                                            label=line["cosSimChar"]))

                logging.info(f"InputExample Data : {len(all_input_examples)}")

                store_dir = os.path.join(args.output_dir, lang, purpose_type)
                os.makedirs(store_dir, mode=0o777, exist_ok=True)
                store_file_name = os.path.join(store_dir, f"{delete_type}.pickle")
                with open(store_file_name, "wb") as p:
                    pickle.dump(all_input_examples, p)
                logging.info(f"Stored pickle data -> {store_file_name}")


    return None


if __name__ == '__main__':
    main()

