
import os
import random
import torch
folder=f"/scratch/obashom.sps.iitmandi/maps/mapchunks"
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import numpy as np
import os
sensit=np.array([1.31,1.15,1.78,1.91,4.66,7.99])

class MapChunkDataset(Dataset):
    def __init__(self, data_dir,k, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing k_* folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.k_folder = os.path.join(data_dir,f'k_{k}')
        self.transform = transform
        self.num_indices = 1000  # Assuming indices 0 to 1000

    def __len__(self):
        return len(os.listdir(self.k_folder))

    def __getitem__(self, idx):
        """
        Loads the data for a given index across all k folders.

        Args:
          idx: (int) index from 0 to len(dataset). This idx will have k index and idx inside folder

        Returns:
          torch.Tensor:  A tensor containing the stacked data, shape: (6, 1024, 1024).
        """

        k_folder = self.k_folder
        in_path = os.path.join(k_folder, 'in', f'{idx}.npy')
        out_path = os.path.join(k_folder, 'out', f'{idx}.npy')
        emission_folder = os.path.join(k_folder, 'emissions')
        var_path = os.path.join(k_folder, 'var', 'variance.npy')


        data_list = []
        var_arrorig = np.load(var_path)
        maxvar=np.max(var_arr)
        in_arr = np.load(in_path)
        for i in range(6):
            emission_arr = np.load(os.path.join(emission_folder, f'{i}.npy'))
            var_arr=var_arrorig*(sensit[i]/maxvar)
            noise = np.random.normal(0, var_arr, size=in_arr.shape) #Gaussian noise with variance given by var_arr
            data_list.append(in_arr + emission_arr + noise)


        stacked_data = np.stack(data_list)
        stacked_data = torch.from_numpy(stacked_data).float() # Convert to torch float tensor
        
        out_arr = np.load(out_path)
        out_arr = torch.from_numpy(out_arr).float()
        
        if self.transform:
            stacked_data = self.transform(stacked_data)

        return stacked_data, out_arr
        

if __name__ == '__main__':
    # Example usage:
    data_dir = "/scratch/obashom.sps.iitmandi/maps/mapchunks"  # Replace with your actual path

    
    dataset = MapChunkDataset(data_dir,0)

    #Example
    first_item, out_item = dataset[0]

    print("Shape of the first element of dataset")
    print(first_item.shape)  # Should be torch.Size([6, 1024, 1024])
    print("Shape of out data")
    print(out_item.shape) # Should be torch.Size([1024, 1024])

    # You can now use this with a DataLoader to iterate over it
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch_idx, (data,out) in enumerate(dataloader):
        print(f"Batch {batch_idx+1} data shape:", data.shape)
        print(f"Batch {batch_idx+1} out shape:", out.shape)
        break # printing only first batch
    