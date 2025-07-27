import os
import json
from typing import List, Dict
from tqdm import tqdm

from datasets import load_dataset


BASE_PATH = "/home/tangwenhao/Workspace/ARC"

def create_dataset_list(data_path: str) -> List[Dict[str, List[List[int]]]]:
    """
    Create a JSON file from the dataset.
    """
    dataset = []
    for name in tqdm(os.listdir(data_path), desc="Loading dataset"):
        if name.endswith('.json'):
            with open(os.path.join(data_path, name), 'r') as f:
                data = json.load(f)
                train_data = data['train']
                test_data = data['test']
            dataset_item = {}
            for i, item in enumerate(train_data):
                dataset_item[f'train_input_{i}'] = item['input']
                dataset_item[f'train_output_{i}'] = item['output']
            for i, item in enumerate(test_data):
                dataset_item[f'test_input_{i}'] = item['input']
                dataset_item[f'test_output_{i}'] = item['output']
            dataset.append(dataset_item)
    return dataset


if __name__ == "__main__":
    # ========== Create dataset ==========
    dataset_train = create_dataset_list(os.path.join(BASE_PATH, "data", "training"))
    with open(os.path.join(BASE_PATH, "train.json"), 'w') as f:
        json.dump(dataset_train, f)
    
    dataset_eval = create_dataset_list(os.path.join(BASE_PATH, "data", "evaluation"))
    with open(os.path.join(BASE_PATH, "eval.json"), 'w') as f:
        json.dump(dataset_eval, f)

    # ========== Load dataset (Example usage) ==========
    file_paths = {
        'train': os.path.join(BASE_PATH, "train.json"),
        'eval': os.path.join(BASE_PATH, "eval.json")
    }

    dataset = load_dataset('json', data_files=file_paths)
    print(f"Dataset loaded with {len(dataset['train'])} training examples and {len(dataset['eval'])} evaluation examples.")
    breakpoint()  # For debugging purposes, can be removed later
    print("Dataset creation completed.")
