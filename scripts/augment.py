import os
import json
import numpy as np
from augment_utils import AUGMENTATIONS
from tqdm import tqdm


base_path = "/home/tangwenhao/Workspace/ARC"

def augment_data(version: int, split: str, name: str):
    """
    Augment the data for a given episode.
    """
    input_path = os.path.join(base_path, f"ARC-AGI-{version}", "data", split, f"{name}.json")
    output_dir = os.path.join(base_path, "data", split)
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r') as f:
        data = json.load(f)
        train_data = data['train']
        test_data = data['test']

    for i in range(len(AUGMENTATIONS)):
        augment_train_data = []
        augment_test_data = []
        for item in train_data:
            augmented_input = AUGMENTATIONS[i](np.array(item['input'])).tolist()
            augmented_output = AUGMENTATIONS[i](np.array(item['output'])).tolist()
            augment_train_data.append({
                'input': augmented_input,
                'output': augmented_output
            })

        for item in test_data:
            augmented_input = AUGMENTATIONS[i](np.array(item['input'])).tolist()
            augmented_output = AUGMENTATIONS[i](np.array(item['output'])).tolist()
            augment_test_data.append({
                'input': augmented_input,
                'output': augmented_output
            })

        output_path = os.path.join(output_dir, f"{name}_{i:02d}.json")
        with open(output_path, 'w') as f:
            json.dump({
                'train': augment_train_data,
                'test': augment_test_data
            }, f)


if __name__ == "__main__":
    version = 2
    split = "training"
    
    name_list = [
        name.split('.')[0] for name in os.listdir(os.path.join(base_path, f"ARC-AGI-{version}", "data", split))
        if name.endswith('.json')
    ]
    for name in tqdm(name_list, desc="Augmenting data"):
        if os.path.exists(os.path.join(base_path, "data", split, f"{name}_00.json")):
            continue
        augment_data(version, split, name)
    print("Data augmentation completed.")
        