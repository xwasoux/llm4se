import json
import sys
import glob
import pickle

from sentence_transformers import InputExample


def get_path_list(path_name):
    condition = f'{path_name}/*.json'
    return glob.glob(condition)


def eval_cos_sim_label(original_code_count, node_deletion_count):
    '''
    ratio : 0 to 1 (0:far)
    output : float number from -1 to 1
    -1 -> far
    1  -> near
    '''
    ratio = 1 - (node_deletion_count / original_code_count)
    return round((ratio*ratio)-1, 3)


def main():
    '''
    perspective

        get json file path

        for file in path:
            file open
            for east in code_list: 
                InputExample(original, east)
                list append

        save to pickle file
    '''

    dataset_path = get_path_list("SentenceCodeBERT/data/semantic_east_sim/semantic_east_sim_data")

    all_input_examples = []
    for path in dataset_path:
        with open(path, 'r') as f:
            json_data = json.load(f)

        original_code = json_data["original_code"]
        original_code_nodes_count = json_data["num_of_nodes"]
        file_name = json_data["filename"]

        each_code_list = []
        for num, edited_code_info in enumerate(json_data["edited_code"]):
            east_code = edited_code_info["code"]
            node_delete_count = edited_code_info["count"]
            cos_sim_ast2east = eval_cos_sim_label(original_code_nodes_count, node_delete_count)
            each_code_list.append(InputExample(guid=f'{file_name}-{num}', texts=[original_code, east_code], label=cos_sim_ast2east))

        all_input_examples.append(each_code_list)

    with open("SentenceCodeBERT/data/semantic_east_sim/semantic_east_sim_dataset.pickle", "wb") as p:
        pickle.dump(all_input_examples, p)


if __name__ == '__main__':
    main()
