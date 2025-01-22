import os
import numpy as np
import healpy as hp
from time import perf_counter
import math
import logging
import cProfile
import pstats
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

nside = 1024
output_dir = '/scratch/obashom.sps.iitmandi/maps/mapchunks'
num_maps = 6
k_list = list(range(12))
input_dir = '/scratch/obashom.sps.iitmandi/maps/unpatchedmaps/ems'


def bin2nbit(binary, n):
    while len(binary) < n:
        binary = '0' + binary
    return binary


def save12patches(cmb_map_array, map_index, fl='in'):
    pr = cProfile.Profile()
    pr.enable()

    t1 = perf_counter()
    lo = int(2 * math.log2(nside))
    mpo = np.zeros((nside, nside, len(k_list)), dtype=np.float32)
    logging.debug(f"save12patches: Starting patch creation for map {map_index}, type {fl}")
    for k_idx, k in enumerate(k_list):
        logging.debug(f'{k} , {map_index}')
        pixar = '0' * lo
        kb = bin2nbit(bin(k)[2:], 4)
        pixar = np.array(list(kb + pixar))
        for i in range(nside):
            ib = np.array(list(bin2nbit(bin(i)[2:], int(math.log2(nside)))))
            for index, value in np.ndenumerate(ib):
                pixar[4 + 2 * index[0]] = value
            for j in range(nside):
                jb = np.array(list(bin2nbit(bin(j)[2:], int(math.log2(nside)))))
                for index, value in np.ndenumerate(jb):
                    pixar[5 + 2 * index[0]] = value
                PixNested = int(''.join([str(score) for score in pixar]), 2)
                PixRing = hp.nest2ring(nside, PixNested)
                mpo[i, j, k_idx] = cmb_map_array[PixRing]
    logging.debug(f"save12patches: Patch creation complete for map {map_index}, type {fl}")

    for k_idx, k_value in enumerate(k_list):
        k_dir = os.path.join(output_dir, f"k_{k_value}/{fl}")
        os.makedirs(k_dir, exist_ok=True)
        file_name = os.path.join(k_dir, f"{map_index}.npy")
        np.save(file_name, mpo[:, :, k_idx])
    logging.debug(f'time taken to save map {map_index} of type {fl} is {perf_counter() - t1}')

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logging.debug(f"save12patches: Profile for map {map_index}, type {fl} \n {s.getvalue()}")


def process_single_map(map_filename, map_index, fl):
    logging.debug(f"process_single_map: Starting for map {map_filename}, index {map_index}, type {fl}")
    try:
        map_data = np.load(map_filename)
        save12patches(map_data, map_index, fl)
        logging.debug(f"process_single_map: Completed for map {map_filename}, index {map_index}, type {fl}")
    except Exception as e:
        logging.error(f"process_single_map: Error processing map {map_filename}, index {map_index}, type {fl}: {e}")


def process_maps_parallel(input_dir, num_maps, fl='in'):
    logging.debug("process_maps_parallel: Starting parallel processing")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        map_filenames = [os.path.join(input_dir, f'map_{i}.npy') for i in range(num_maps)]
        map_indices = list(range(num_maps))
        executor.map(process_single_map, map_filenames, map_indices, [fl] * num_maps)
    logging.debug("process_maps_parallel: Finished parallel processing")

if __name__ == '__main__':
    print("Loading and processing data...")
    logging.debug("main: Starting data loading and processing")
    
    mp.set_start_method('spawn')
    process_maps_parallel(input_dir, num_maps, fl='emissions')

    print('all maps done')
    logging.debug("main: Finished all maps processing")