import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import List, Dict
from tqdm import tqdm


base_path = "/home/tangwenhao/Workspace/ARC"

colors = [
    '#000000',  # 0: Black
    '#1075D1',  # 1: Blue
    '#F24438',  # 2: Red
    '#2DBE46',  # 3: Green
    '#FBE82A',  # 4: Yellow
    '#A9A9A9',  # 5: Gray
    '#EB25B0',  # 6: Magenta
    '#F88022',  # 7: Orange
    '#7AD6F5',  # 8: Light Blue
    '#800000'   # 9: Maroon
]
cmap = ListedColormap(colors)
bounds = np.arange(-0.5, 10, 1)
norm = BoundaryNorm(bounds, cmap.N)


def add_edge(ax: plt.Axes, matrix: np.ndarray, edge_color: str = 'w'):
    # Get matrix dimensions
    h, w = matrix.shape

    # Set minor ticks to be on the halfway point between pixels
    ax.set_xticks(np.arange(w+1)-.5, minor=True)
    ax.set_yticks(np.arange(h+1)-.5, minor=True)

    # Add grid lines based on minor ticks
    ax.grid(which='minor', color=edge_color, linestyle='-', linewidth=1)

    # Hide major tick labels
    ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)


def visualize(version: int, split: str, name: str):
    # Load the data from the JSON file
    with open(os.path.join(base_path, f"ARC-AGI-{version}", "data", split, f"{name}.json"), "r") as f:
        data = json.load(f)

    train_data = data.get('train', [])
    test_data = data.get('test', [])

    num_shots = len(train_data)

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(num_shots, 4, figure=fig)
    fig.suptitle(f"ARC-AGI-{version} - {split.capitalize()} - {name}", fontsize=16)

    for i, shot in enumerate(train_data):
        ax1 = fig.add_subplot(gs[i, 0])
        input_data = np.array(shot['input'])
        ax1.imshow(input_data, cmap=cmap, norm=norm, aspect='equal')
        add_edge(ax1, input_data)
        ax1.set_title(f"Input {i+1}")

        ax2 = fig.add_subplot(gs[i, 1])
        output_data = np.array(shot['output'])
        ax2.imshow(output_data, cmap=cmap, norm=norm, aspect='equal')
        add_edge(ax2, output_data)
        ax2.set_title(f"Output {i+1}")
    
    for j, shot in enumerate(test_data):
        ax3 = fig.add_subplot(gs[j, 2])
        test_input_data = np.array(shot['input'])
        ax3.imshow(test_input_data, cmap=cmap, norm=norm, aspect='equal')
        add_edge(ax3, test_input_data)
        ax3.set_title(f"Test Input {j+1}")

        ax4 = fig.add_subplot(gs[j, 3])
        test_output_data = np.array(shot['output'])
        ax4.imshow(test_output_data, cmap=cmap, norm=norm, aspect='equal')
        add_edge(ax4, test_output_data)
        ax4.set_title(f"Test Output {j+1}")


    output_path = os.path.join(base_path, "scripts", "visualizations", f"v{version}", split, f"{name}.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # visualize(2, "training", "0a938d79")
    version = 2
    split = "training"
    names = [filename.split(".")[0] for filename in os.listdir(os.path.join(base_path, f"ARC-AGI-{version}", "data", split)) if filename.endswith(".json")]
    for name in tqdm(names, desc=f"Visualizing {version} {split}"):
        visualize(version, split, name)
