import os
import re
import sys
import csv
import gzip
import logging
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime

from git import Repo
from tree_sitter import Language, Parser
from parser import remove_comments_and_docstrings, get_functions

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO
                    )

parsers = {}

def get_repo_name(args: argparse) -> str:
    res = re.findall(r'(https://github.com/|git@github.com:)(.*)/(.*).git', args.girhub_repo)
    user_name = res[0][1]
    repo_name = res[0][2]
    return user_name, repo_name

def dir_manage(user_name: str, repo_name: str) -> str:
    clone_dir = os.path.join(os.path.dirname(__file__), "clone", user_name, repo_name)
    os.makedirs(clone_dir, exist_ok=True)
    deploy_dir = os.path.join(os.path.dirname(__file__), "inputs", user_name, repo_name)
    os.makedirs(deploy_dir, exist_ok=True)
    return clone_dir, deploy_dir
    
def retreve_target_files(args: argparse, clone_dir: str) -> list:
    return glob(os.path.join(clone_dir, "**", f"*.{args.language}"), recursive=True)

def remove_abs_path(file_path: str, user_repo_name:str) -> str:
    ancectors = re.match(rf'.*({user_repo_name})/', file_path)
    return file_path.replace(ancectors.group(0), "")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--girhub_repo", type=str)
    parser.add_argument("--branch", type=str)
    parser.add_argument("--language", type=str)

    args = parser.parse_args()

    user_name, repo_name = get_repo_name(args)
    user_repo_name = f"{user_name}/{repo_name}"

    clone_dir, deploy_dir = dir_manage(user_name, repo_name)

    ## Start clone
    # repo = Repo.clone_from(args.girhub_repo, clone_dir, branch=args.branch)
    
    ## Retreve target files
    target_file_paths = retreve_target_files(args, clone_dir)

    ## Parse
    LANGUAGE = Language(os.path.join("parser", "my-languages.so"), args.language)
    parser = Parser()
    parser.set_language(LANGUAGE)

    fnc_list = []

    for file in tqdm(target_file_paths):
        ## Read file
        with open(file) as f:
            code = f.read()
        
        ## parse
        pure_code = remove_comments_and_docstrings(code, args.language)
        tree = parser.parse(bytes(pure_code, "utf8"))
        fnc_trees = get_functions(tree)

        for fnc in fnc_trees:
            fnc_info = {}
            fnc_info["repo"] = user_repo_name
            fnc_info["path"] = remove_abs_path(file, user_repo_name)
            fnc_info["original_string"] = code
            fnc_info["language"] = args.language
            fnc_info["cleaned_code"] = fnc.text.decode()
        
            fnc_list.append(fnc_info)
    
    df = pd.DataFrame(fnc_list)
    path = os.path.join(deploy_dir, "inputs.jsonl")
    df.to_json(path, force_ascii=False, lines=True, orient='records')
    logging.info(f"Finish writing: {path}")
        
    logging.info(f"\nExtract Results\n\tTarget files: {len(target_file_paths)}\n\tFunctions: {len(fnc_list)}")

    return None


if __name__ == "__main__":
    main()