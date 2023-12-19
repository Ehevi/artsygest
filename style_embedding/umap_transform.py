from time import time
from pathlib import Path

import numpy as np
import torch
import umap
import pickle

from tqdm import tqdm

EMBEDDING_PATH = Path("/media/pawel/DATA/iwisum/results/style_embeddings")
OUTPUT_PATH = Path("/media/pawel/DATA/iwisum/results/umap_embeddings")

BATCH_SIZE = 10

def save_pickle(data, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)
    print(f'Pickle file saved to {filepath}')


def load_tensors(filepaths: list) -> list[torch.Tensor]:
    return [torch.load(file) for file in filepaths]

def concatenate_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0)

def concatenate_nd_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrays, axis=0)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy().astype('float32')

def umap_fit(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(tensor)
    return umap.UMAP().fit(arr)

def save_numpy_array(arr: np.ndarray, filepath: Path) -> None:
    np.save(filepath, arr)
    print(f'Numpy array saved to {filepath}')

def fit_style_embedding(layer_name: str, filepaths: list) -> tuple[np.ndarray, str]:
    print(f"Fitting style embedding for {layer_name}")

    tensors = load_tensors(filepaths)
    print(f'Loaded {len(tensors)} tensors')

    stacked_tensor = concatenate_tensors(tensors)
    print(f'Stacked tensor shape: {stacked_tensor.shape}')

    umap_fitted = umap_fit(stacked_tensor)
    
    # Save UMAP fitted to a file
    umap_filepath = OUTPUT_PATH / f'{layer_name}_umap.npy'
    save_pickle(umap_fitted, umap_filepath)

    return umap_fitted, layer_name

def transform_style_embedding(umap_fitted: np.ndarray, filepaths: list) -> np.ndarray:
    print("Transforming style embedding")

    tensors = load_tensors(filepaths)
    print(f'Loaded {len(tensors)} tensors')

    transformed_arrays = [umap_fitted.transform(tensor_to_numpy(tensor)) for tensor in tqdm(tensors)]
    print(f'Transformed {len(transformed_arrays)} tensors')

    result_array = concatenate_nd_arrays(transformed_arrays)
    print(f'Result array shape: {result_array.shape}')

    return result_array

def main():
    layer_names = ["relu1", "relu2", "relu3", "relu4", "relu5"]
    for layer_name in layer_names:
        print(f'Starting {layer_name}')

        files_to_fit = list(EMBEDDING_PATH.rglob(f'**/*{layer_name}*_0.pt'))   # we take only 100 samples for each layer for each category
        files_to_transform = list(EMBEDDING_PATH.rglob(f'**/*{layer_name}*'))
        files_to_transform_batched = [files_to_transform[i:i+BATCH_SIZE] for i in range(0, len(files_to_transform), BATCH_SIZE)]
        print(f'Found {len(files_to_fit)} files to fit and {len(files_to_transform)} files to transform')
        
        print("Fitting and transforming style embedding")
        start = time()
        umap_fitted, _ = fit_style_embedding(layer_name, files_to_fit)
        print(f'Finished fitting in {time() - start} seconds')

        for i, files in enumerate(files_to_transform_batched):
            print(f'Transforming batch number {i+1}')
            start = time()
            umap_array = transform_style_embedding(umap_fitted, files)
            print(f'Finished transforming in {time() - start} seconds')
            
            out_name = f'{layer_name}_umap_{i}.npy'
            out_name = OUTPUT_PATH / out_name
            save_numpy_array(umap_array, out_name)
            print(f'Saved {out_name}. Shape: {umap_array.shape}. Size: {umap_array.nbytes / 1e6} MB')

if __name__ == '__main__':
    main()