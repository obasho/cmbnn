import os
import numpy as np
import logging

fpth = '/scratch/obashom.sps.iitmandi/maps/unpatchedmaps'
fl = 'ems'
k_dir = os.path.join(fpth, fl)
os.makedirs(k_dir, exist_ok=True)

logging.debug("main: Starting data loading and processing")

# Load data
data_filename = "/scratch/obashom.sps.iitmandi/maps/Con_em_maps.npy"
num_maps = 6

data = np.load(data_filename)

# Save individual maps
for i in range(num_maps):
    fname = os.path.join(k_dir, f'map_{i}.npy')
    np.save(fname, data[:, i])  # Corrected the argument order
