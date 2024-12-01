
import numpy as np
import torch
import os
import random


folder=f"/scratch/obashom.sps.iitmandi/cmb/generation/"




def load_data(filepath):
    data = np.load(filepath,allow_pickle=True).item() 
    return data['input'], data['output']
def load_random_pair(d, nside):
    folderd = folder+f"maps/{nside}{d}/"
    
    files = os.listdir(folderd)
    if not files:
        raise ValueError(f"No files found in the folder {folder}")
    
    idx = random.choice(files)
    inpath = os.path.join(folderd, idx)
    
    input_array, output_array = load_data(inpath)
    
    input_tensor = torch.tensor(input_array, dtype=torch.float32)
    output_tensor = torch.tensor(output_array, dtype=torch.float32)
    
    return input_tensor, output_tensor
