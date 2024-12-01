import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import healpy as hp
from time import perf_counter
import math
import os
import subprocess
import numpy as np
from astropy.io import fits
import os

folder = f"/scratch/obashom.sps.iitmandi/cmb/generation/"
k_list=[6,7,8,9,10,11] #k values to be generated
frequency=[70,100,143,217,353,545]
fwhm=np.array([13.31,9.66,7.27,5.01,4.86,4.84])/60 
sensit=np.array([1.31,1.15,1.78,1.91,4.66,7.99])
cl_file = folder+"cl.fits"  # Path to the power spectrum file
alm_file =folder+ "alms.fits"    # Temporary file for a_lm output
map_file =folder+ "cmb_map.fits" # Temporary file for the CMB map output
nside = 1024                      # Desired nside parameter for the output map
nlmax = 2506                      # Maximum order of l
polarisation = False              # Whether to generate polarisation (IQU) or intensity only (I)
full_ps = False                   # Full power spectrum or partial (TT, GG, CC, TG only)
double_precision = True          # Single or double precision for a_lm and maps
rand_seed = 1234                  # Random seed for reproducibility
input_map_file =folder+ "full_map.fits"   # Path to the input HEALPix map file
output_smoothed_file = folder+"smoothed_map.fits"  # Path to save the smoothed map
polarisation = False                        # Set to True if the input map has polarisation data (IQU), else False for intensity only
weighted = False                            # Set to True to use weighted quadrature
iter_order = 0                              # Number of iterations (usually 0 for standard)


def smooth_healpix_map(infile, outfile, fwhm_arcmin, nlmax, polarisation=False, weighted=False, iter_order=0, double_precision=False):
    smoothing_command = [
        "smoothing_cxx",
        f"fwhm_arcmin={fwhm_arcmin}",
        f"nlmax={nlmax}",
        f"infile={infile}",
        f"outfile={outfile}",
        f"polarisation={'true' if polarisation else 'false'}",
        f"iter_order={iter_order}",
        f"double_precision={'true' if double_precision else 'false'}"
    ]

    # Execute the command
    subprocess.run(smoothing_command,stdout=subprocess.DEVNULL, check=True)


    smoothed_array=hp.fitsfunc.read_map(outfile,nest=False)
    # Clean up by removing the FITS file
    os.remove(outfile)
    os.remove(infile)
    return smoothed_array

def bin2nbit(binary,n):
  while(len(binary)<n):
    binary='0'+binary
  return binary

def generate_cmb_map(cl_file, alm_file, map_file, nside, nlmax, fwhm_arcmin, polarisation=False, full_ps=False, double_precision=False, rand_seed=1234):
    # Run syn_alm_cxx to generate a_lm
    syn_alm_command = [
        "syn_alm_cxx",
        f"nlmax={nlmax}",
        f"infile={cl_file}",
        f"outfile={alm_file}",
        f"rand_seed={rand_seed}",
        f"polarisation={'true' if polarisation else 'false'}",
        f"full_ps={'true' if full_ps else 'false'}",
        f"double_precision={'true' if double_precision else 'false'}"
    ]
    subprocess.run(syn_alm_command,stdout=subprocess.DEVNULL, check=True)

    # Run alm2map_cxx to convert a_lm to a HEALPix map
    alm2map_command = [
        "alm2map_cxx",
        f"nlmax={nlmax}",
        f"infile={alm_file}",
        f"outfile={map_file}",
        f"nside={nside}",
        f"polarisation={'true' if polarisation else 'false'}",
        f"double_precision={'true' if double_precision else 'false'}"
    ]
    subprocess.run(alm2map_command,stdout=subprocess.DEVNULL, check=True)

    cmb_map_array=hp.fitsfunc.read_map(map_file,nest=False)
    os.remove(alm_file)
    os.remove(map_file)

    return cmb_map_array


def getpatches(folder, cmb_map_array,  k_list,tre):
    t1 = perf_counter()
    lo = int(2 * math.log2(nside))
    mp = np.zeros((nside, nside, 6, len(k_list)), dtype=np.float32)
    mpo = np.zeros((nside, nside, len(k_list)), dtype=np.float32)
    
    for r in range(6):
        fm = hp.fitsfunc.read_map(folder + f'emission_map/emission_map_0_{r}.fits')
        fmap = fm + cmb_map_array
        mf = fmap + np.random.normal(scale=sensit[r], size=len(fmap))

        # Step 2: Save the combined map as a FITS file
        input_map_file = f"temp_map_{r}_{tre}.fits"
        hp.fitsfunc.write_map(input_map_file, mf)
        
        # Step 3: Smooth the map
        output_smoothed_file = f"smoothed_map_{r}_{tre}.fits"
        smoothed_map_array = smooth_healpix_map(
            infile=input_map_file,
            outfile=output_smoothed_file,
            fwhm_arcmin=fwhm[r],
            nlmax=nlmax,
            polarisation=polarisation,
            weighted=weighted,
            iter_order=iter_order,
            double_precision=double_precision
        )

        # Step 4: Process pixels and store in arrays
        for k in k_list:
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
                    mp[i, j, r, k_list.index(k)] = smoothed_map_array[PixRing]
                    if r == 4:
                        mpo[i, j, k_list.index(k)] = cmb_map_array[PixRing]

    return mp, mpo ,perf_counter()-t1

def genmap(n):
    t0=perf_counter()
    for k in k_list:
        folderk = folder + f"maps/{nside}{k}/"
        os.makedirs(folderk, exist_ok=True)
        
    alm_file = folder+f"alms{n}.fits"  # Temporary file for a_lm output
    map_file = folder+f"cmb_map{n}.fits"
    
    cmb_map_array = generate_cmb_map(
        cl_file=cl_file, alm_file=alm_file, map_file=map_file, 
        nside=nside, nlmax=nlmax, fwhm_arcmin=fwhm[0], 
        polarisation=polarisation, full_ps=full_ps, 
        double_precision=double_precision, rand_seed=rand_seed
    )
    
    mp, mpo,t = getpatches(folder, cmb_map_array, k_list,n)
    
    for k in k_list:
        folderk = folder + f"maps/{nside}{k}/"
        np.save(os.path.join(folderk, f"data_{n}.npy"), {
            'input': mp[:, :, :, k_list.index(k)], 
            'output': mpo[:, :, k_list.index(k)]
        })
    return t,(perf_counter()-t0)


def parallel_genmap(n0, nf, max_workers=8):
    """Parallel execution of genmap using ProcessPoolExecutor."""
    t3=perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(genmap, n) for n in range(n0, nf)]

        for future in as_completed(futures):
            try:
                t, elapsed_time = future.result()
            except Exception as e:
                print(f"Error in processing: {e}")
    return perf_counter()-t3

def run_in_batches(n0, nf, batch_size=24):
    current = n0
    while current < nf:
        next_batch = min(current + batch_size, nf)
        t5=parallel_genmap(current, next_batch, max_workers=batch_size)
        current = next_batch
        print(f"time for processing {batch_size} samples is {t5/60}")

run_in_batches(0,1200)