import os
import json
from collections import Counter

base_path = "/home/tangwenhao/Workspace/ARC"

def record_stats(version: int, split: str, name: str):
    with open(os.path.join(base_path, f"ARC-AGI-{version}", "data", split, f"{name}.json"), "r") as f:
        data = json.load(f)
        train_data = data["train"]

    num_shots = len(train_data)
    input_shape = (len(train_data[0]["input"]), len(train_data[0]["input"][0]))
    output_shape = (len(train_data[0]["output"]), len(train_data[0]["output"][0]))
    is_shape_equal = input_shape == output_shape

    return {
        "num_shots": num_shots,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "is_shape_equal": is_shape_equal
    }



if __name__ == "__main__":

    for version in [1, 2]:
        for split in ["training", "evaluation"]:
            stats = {}
            num_shots = []
            is_shape_equal = []

            for filename in os.listdir(os.path.join(base_path, f"ARC-AGI-{version}", "data", split)):
                if filename.endswith(".json"):
                    name = filename.split(".")[0]
                    result = record_stats(version, split, name)
                    stats[name] = result
                    num_shots.append(result["num_shots"])
                    is_shape_equal.append(result["is_shape_equal"])
            
            with open(os.path.join(base_path, "scripts", "stats", f"stats_v{version}_{split}.json"), "w") as f:
                json.dump(stats, f, indent=4)

            with open(os.path.join(base_path, "scripts", "stats", f"counts_v{version}_{split}.json"), "w") as f:
                json.dump({
                    "num_shots": dict(Counter(num_shots)),
                    "is_shape_equal": dict(Counter(is_shape_equal))
                }, f, indent=4)
    

