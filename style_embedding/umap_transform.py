from time import time
from pathlib import Path

import numpy as np
import torch
import umap

from tqdm import tqdm

def load_tensors(filepath: Path) -> list[torch.Tensor]:
    return [torch.load(file) for i, file in enumerate(filepath.iterdir()) if i < 14]


def concatenate_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy().astype('float32')

def umap_transform(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(tensor)
    return umap.UMAP().fit_transform(arr)
    
def save_numpy_array(arr: np.ndarray, filepath: Path) -> None:
    np.save(filepath, arr)
    print(f'Numpy array saved to {filepath}')

def transform_style_embedding(layer_name: str) -> np.ndarray:
    filepath = Path(f'sample_vectors/{layer_name}')
    tensors = load_tensors(filepath)
    stacked_tensor = concatenate_tensors(tensors)
    print(f'Shape before transform: {stacked_tensor.shape}')
    return umap_transform(stacked_tensor)
    


def main():
    layer_names = ["relu_1", "relu_2", "relu_3", "relu_4", "relu_5"]
    for layer_name in layer_names:
        print(f'Starting {layer_name}')
        start = time()
        umap_array = transform_style_embedding(layer_name)
        print(f'Finished {layer_name} in {time() - start} seconds')
        save_numpy_array(umap_array, f'{layer_name}_umap.npy')
        print(f'Saved {layer_name}_umap.npy. Shape: {umap_array.shape}. Size: {umap_array.nbytes / 1e6} MB')

if __name__ == '__main__':
    main()