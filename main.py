# /// script
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "transformers",
#   "torch==2.5.1",
# ]
# ///

# @author: x.com/@attentionmech
# name: trunk

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from transformers import AutoModel, AutoModelForCausalLM
import re
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse

def draw_cube(ax, position, size, color):
    x, y, z = position
    s = size
    x -= s / 2
    y -= s / 2

    vertices = np.array([
        [x, y, z], [x + s, y, z], [x + s, y + s, z], [x, y + s, z],
        [x, y, z + s], [x + s, y, z + s], [x + s, y + s, z + s], [x, y + s, z + s]
    ])

    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [1, 2, 6, 5]],
        [vertices[j] for j in [4, 7, 3, 0]]
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.2, alpha=0.9))

def clean_layer_name(layer_name):
    return re.sub(r'\.\d+\.', '.', layer_name)

def generate_architecture(model_name):
    model = AutoModel.from_pretrained(model_name)
    state_dict = model.state_dict()

    cube_list = []
    unique_layers = sorted(set(clean_layer_name(layer) for layer in state_dict.keys()))
    cmap = cm.get_cmap("rainbow", len(unique_layers))
    layer_color_map = {layer: mcolors.to_hex(cmap(i)) for i, layer in enumerate(unique_layers)}

    max_param_size = max(v.numel() for v in state_dict.values())
    min_param_size = min(v.numel() for v in state_dict.values())
    min_size = 0.5
    max_size = 2.0

    for layer, params in state_dict.items():
        param_size = params.numel()
        size = min_size + ((np.log(param_size) - np.log(min_param_size)) / (np.log(max_param_size) - np.log(min_param_size))) * (max_size - min_size)
        clean_name = clean_layer_name(layer)
        color = layer_color_map.get(clean_name, "gray")

        cube_list.append((color, size))

    return cube_list

def main():
    parser = argparse.ArgumentParser(description="Visualize Transformer Model Architecture in 3D")
    parser.add_argument("model", type=str, help="Name of the model to load from Hugging Face")
    parser.add_argument("--zoom", type=float, default=1.0, help="Zoom factor (higher means zoomed out)")
    parser.add_argument("--elev", type=float, default=30, help="Elevation angle (tilt up/down)")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle (rotate left/right)")
    args = parser.parse_args()

    torch.set_default_device("meta")
    
    cube_list = generate_architecture(args.model)

    fig = plt.figure(figsize=(6, 10))
    ax = fig.add_subplot(111, projection='3d')

    z_offset = 0
    for color, size in cube_list:
        draw_cube(ax, (0, 0, z_offset), size, color)
        z_offset += size * 2.2

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.axis('off')

    zoomed_limit = 3 / args.zoom
    ax.set_xlim([-zoomed_limit, zoomed_limit])
    ax.set_ylim([-zoomed_limit, zoomed_limit])
    ax.set_zlim([-0.5, z_offset + 1])

    ax.set_box_aspect([1, 1, 4])  # Maintain aspect ratio
    ax.view_init(elev=args.elev, azim=args.azim)  # Apply rotation

    plt.title(f"{args.model}", fontsize=14, fontweight='bold', pad=20)
    plt.show()

if __name__ == "__main__":
    main()
