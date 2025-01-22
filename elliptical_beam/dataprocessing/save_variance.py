import os
import numpy as np
import healpy as hp
from time import perf_counter
import math

def bin2nbit(binary, n):
    return binary.zfill(n)

def save12patches(variance_array, fl='var'):
    t1 = perf_counter()
    lo = int(2 * math.log2(nside))
    mpo = np.zeros((nside, nside, len(k_list)), dtype=np.float32)
    for k_idx, k in enumerate(k_list):
        pixar = np.array(list(bin2nbit(bin(k)[2:], 4) + '0' * lo))
        for i in range(nside):
            ib = np.array(list(bin2nbit(bin(i)[2:], int(math.log2(nside)))))
            for index, value in enumerate(ib):
                pixar[4 + 2 * index] = value
            for j in range(nside):
                jb = np.array(list(bin2nbit(bin(j)[2:], int(math.log2(nside)))))
                for index, value in enumerate(jb):
                    pixar[5 + 2 * index] = value
                PixNested = int(''.join(pixar), 2)
                PixRing = hp.nest2ring(nside, PixNested)
                mpo[i, j, k_idx] = variance_array[PixRing]

    for k_idx, k_value in enumerate(k_list):
        k_dir = os.path.join(output_dir, f"k_{k_value}/{fl}")
        try:
            os.makedirs(k_dir, exist_ok=True)
            file_name = os.path.join(k_dir, "variance.npy")
            np.save(file_name, mpo[:, :, k_idx])
        except Exception as e:
            print(f"Error saving file for k={k_value}: {e}")

nside = 1024
output_dir = '/scratch/obashom.sps.iitmandi/maps/mapchunks'
k_list = list(range(12))
inputmap = '/scratch/obashom.sps.iitmandi/maps/scan_count.npy'

if not os.path.exists(inputmap):
    raise FileNotFoundError(f"Input map file {inputmap} not found.")

data = np.load(inputmap)
safe_data = np.maximum(data, 1e-12)  # Replace small/zero values with a safe lower bound
variance = np.sqrt(1 / safe_data[:,1])
print(variance.shape)
save12patches(variance)
logger.info("Processing completed.")
