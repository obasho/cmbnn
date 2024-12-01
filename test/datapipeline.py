
import numpy as np
import torch
import os
import random


folder=f"/scratch/obashom.sps.iitmandi/cmb/generation/"
def load_random_pair(d, nside, batch_size=32):
    # Construct the folder path using the given d and nside values
    folderd = folder + f"maps/{nside}{d}/"
    
    # List all files in the folder
    files = os.listdir(folderd)
    if not files:
        raise ValueError(f"No files found in the folder {folderd}")
    
    # Randomly select a batch of files
    selected_files = random.sample(files, min(batch_size, len(files)))
    
    input_tensors = []
    output_tensors = []
    
    # Load data from each selected file and convert to PyTorch tensors
    for filename in selected_files:
        inpath = os.path.join(folderd, filename)
        input_array, output_array = load_data(inpath)
        
        # Convert numpy arrays to PyTorch tensors
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        output_tensor = torch.tensor(output_array, dtype=torch.float32)
        
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    # Stack tensors into batches
    batch_input = torch.stack(input_tensors)
    batch_output = torch.stack(output_tensors)
    
    return batch_input, batch_output


def load_data(filepath):
    # Example implementation; replace with actual data loading logic
    data = np.load(filepath,allow_pickle=True).item()  # Assumes the file contains numpy arrays
    return data['input'], data['output']


