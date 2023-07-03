import re
import os
import glob
import logging
import pickle
import Levenshtein
from anytree import Node, RenderTree
from sentence_transformers import InputExample

s1 = Node("step01")
s2 = Node("step02", parent=s1)
s3 = Node("step03", parent=s2)
s4 = Node("step04", parent=s3)
s6 = Node("step06", parent=s4)
s7 = Node("step07", parent=s6)
s8 = Node("step08", parent=s7)
s9 = Node("step09", parent=s8)

s10 = Node("step10", parent=s9)
s11 = Node("step11", parent=s10)
s12 = Node("step12", parent=s11)
s13 = Node("step13", parent=s12)
s14 = Node("step14", parent=s13)
s16 = Node("step16", parent=s14)
s17 = Node("step17", parent=s16)
s18 = Node("step18", parent=s17)
s19 = Node("step19", parent=s18)

s20 = Node("step20", parent=s19)
s21 = Node("step21", parent=s20)
s22 = Node("step22", parent=s21)

s23 = Node("step23")
s24 = Node("step24", parent=s23)
s26 = Node("step26", parent=s24)
s27 = Node("step27", parent=s23)
s28 = Node("step28", parent=s23)
s29 = Node("step29", parent=s23)

s33 = Node("step33", parent=s29)

s59 = Node("step59")
s60 = Node("step60", parent=s59)

def make_code_pair(root):
    pair_lists = []
    all_nodes = list(root.descendants) 

    source = root

    while all_nodes:
        source_descendants = list(source.descendants)
        while source_descendants:
            target = source_descendants.pop(0)
            pair_lists.append([source, target])
        source = all_nodes.pop(0)
        
    return pair_lists


def get_code_pairs(diff_pair_list, base_path):
    code_pair_dict = {}

    for pair in diff_pair_list:
        code_pair = []
        for code in pair:
            repo_path = glob.glob(base_path + f"/{code.name}.py")
            with open(repo_path[0]) as f:
                char = f.read()
            code_pair.append(char)
        code_pair_dict[f"{pair[0].name}_{pair[1].name}"] = code_pair

    return code_pair_dict

def diff_to_cossim(leven_dist):
    return 1/(1+leven_dist)


def formatting_data(code_diff_pairs):
    sbert_input_data = []
    for index, pair in code_diff_pairs.items():
        source = pair[0]
        target = pair[1]
        distance = Levenshtein.distance(source, target)
        cos_sim = diff_to_cossim(distance)
        sbert_input_data.append(InputExample(guid=index, texts=[source, target], label=cos_sim))
    return sbert_input_data


def main():
    '''
    the way how to make datasets.
    1. search hand made editing process.
    2. create diff pair using list.
    3. culculate diff distance between some code pairs.
    3. store the data using pickle.
    '''
    logging.basicConfig(format='\n%(asctime)s $\n %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    path = "deep-learning-from-scratch-3/steps"

    diff_pair_list = []
    diff_pair_list.extend(make_code_pair(s1))
    
    code_diff_pairs = get_code_pairs(diff_pair_list, path)
    sbert_input_data = formatting_data(code_diff_pairs)
    logging.info(f"You can make dataset! : size - {len(sbert_input_data)}")
    
    with open("../SentenceCodeBERT/data/semantic_diff_sim/dezero_diff_dataset.pickle", "wb") as p:
        pickle.dump(sbert_input_data, p)

    return None

if __name__ == "__main__":
    main()