import torch
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
import gc
import os
import subprocess
import healpy as hp
import numpy as np
from models import Unet, Discriminator
from datapipeline import load_random_pair
def bin2nbit(binary,n):
  while(len(binary)<n):
    binary='0'+binary
  return binary

torch.autograd.set_detect_anomaly(True)
checkpoint_dir = '/scratch/obashom.sps.iitmandi/cmb/train/checkpoints'
input_map_file = "/scratch/obashom.sps.iitmandi/cmb/test/map.fits"   # Path to the input HEALPix map file

k_list=[]#list of k values

def initialize_models(k, nside=1024,device=None,rank=0):
    log_dir = f'./tensorboard_logs_{k}'
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize models on the default device
    inunet = Unet(nside, 6, 1).to(device)
    indiscriminator = Discriminator(nside, 6, 1).to(device)


    # Optimizers
    unet_optimizer = optim.Adam(inunet.parameters(), lr=2e-6)
    discriminator_optimizer = optim.Adam(indiscriminator.parameters(), lr=2e-7, betas=(0.5, 0.999))

    # Checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    unet_checkpoint_path = os.path.join(checkpoint_dir, f'{nside}{k}main.pth')
    indis_checkpoint_path = os.path.join(checkpoint_dir, f'{nside}{k}indis.pth')

    # Load checkpoints if they exist
    if os.path.exists(unet_checkpoint_path):
        inunet.load_state_dict(torch.load(unet_checkpoint_path, map_location=device))

    if os.path.exists(indis_checkpoint_path):
        indiscriminator.load_state_dict(torch.load(indis_checkpoint_path, map_location=device))

    return inunet, indiscriminator, unet_optimizer, discriminator_optimizer, writer, checkpoint_dir




def compute_temperature_power_spectrum(
    map_file, lmax=3072, nmmax=None, remove_monopole=False,
    double_precision=False, outfile="cl.fits"
):
    """
    Compute the temperature power spectrum from a CMB map using anafast_cxx and return it as a NumPy array.

    Parameters:
    - map_file: Path to the input CMB map file.
    - lmax: Maximum multipole moment to compute.
    - nmmax: Maximum m-mode to include (defaults to lmax if not specified).
    - remove_monopole: Whether to remove the monopole.
    - double_precision: Whether to use double precision for computations.
    - outfile: Path to the output FITS file.

    Returns:
    - A NumPy array containing the temperature power spectrum.
    """
    
    # Specify the absolute path for the anafast_cxx executable
    anafast_cxx_path = "/home/obashom.sps.iitmandi/.conda/envs/myenv39/bin/anafast_cxx"
    
    # Check if the executable exists
    if not os.path.exists(anafast_cxx_path):
        print(f"Error: anafast_cxx not found at {anafast_cxx_path}")
        return None

    # Set default value for nmmax if not provided
    if nmmax is None:
        nmmax = lmax
    
    if os.path.exists(outfile):
            os.remove(outfile)
    
    
    try:
        result = subprocess.run([
    "/home/obashom.sps.iitmandi/.conda/envs/myenv39/bin/anafast_cxx",
    f"nlmax={lmax}", f"nmmax={nmmax}",
    f"infile={map_file}", f"outfile={outfile}",
    f"remove_monopole={'true' if remove_monopole else 'false'}",
    f"double_precision={'true' if double_precision else 'false'}",
    "polarisation=false", "overwrite=True"
], check=True)

        print("Anafast execution successful!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running anafast_cxx: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: anafast_cxx command not found. Please check your environment setup.")
        return None

    # Check if the output file was created
    if not os.path.exists(outfile):
        print(f"Output file {outfile} not found!")
        return None
    
    # Read the FITS file and return the power spectrum as a NumPy array
    try:
        
        cl_array = hp.read_cl(outfile)
        print("Successfully read power spectrum from FITS file.")
        return cl_array
    except Exception as e:
        print(f"Error reading FITS file with healpy: {e}")
        return None


    


def put_patch_on_map(nside,k,mp,map):
    lo = int(2 * math.log2(nside))
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
            map[PixRing]=mp[i, j]
    return map

def predict(no_test_samples, k_list,nside=1024):
    t0 = perf_counter()
    gc.collect()
    torch.manual_seed(42)  # Optional: for reproducibility

    # Initialize models for each k
    inunets = []
    for k in k_list:
        inunet, _, _, _, writer, checkpoint_dir = initialize_models(k, nside, 'cpu')
        inunet.eval()  # Set model to evaluation mode
        inunets.append(inunet)
    gc.collect()

    # Run predictions
    cl=np.zeros(3072)
    for _ in range(no_test_samples):
        mapd= np.zeros(12*nside*nside)  # import numpy as np
        for ki, k in enumerate(k_list):
            inp, out = load_random_pair(k, nside, batch_size=1)
            
            input_image = inp.clone().detach().float().permute(0, 3, 1, 2)
            target = out.clone().detach().unsqueeze(0).float()

            # Get prediction from the model
            with torch.no_grad():
                geny_output = inunets[ki](input_image)
            geny_output = geny_output.squeeze()

            mapd=put_patch_on_map(nside,k,geny_output,mapd)
        
        if os.path.exists(input_map_file):
            os.remove(input_map_file)
        hp.fitsfunc.write_map(input_map_file, mapd, nest=False)
        cl += compute_temperature_power_spectrum(input_map_file)[1:]

        gc.collect()
        print(f"step {_} time ellapsed : {(perf_counter()-t0)/60} min")

    cl*=(12/(no_test_samples*len(k_list)))
    np.savetxt("cl.txt",cl)
            
            
            
            
predict(1,k_list)



    
    







