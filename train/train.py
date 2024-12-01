import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from time import perf_counter
import gc
import os
import numpy as np
import time
from datapipeline import load_random_pair
from models import Unet, Discriminator
from utils import generator_loss, discriminator_loss

torch.autograd.set_detect_anomaly(True)


def initialize_models(k, nside=1024,device=None,rank=0):
    log_dir = f'./tensorboard_logs_{k}'
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize models on the default device
    inunet = Unet(nside, 6, 1).to(device)
    indiscriminator = Discriminator(nside, 6, 1).to(device)


    # Optimizers
    unet_optimizer = optim.Adam(inunet.parameters(), lr=2e-5)
    discriminator_optimizer = optim.Adam(indiscriminator.parameters(), lr=2e-7, betas=(0.5, 0.999))

    # Checkpoints
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    unet_checkpoint_path = os.path.join(checkpoint_dir, f'{nside}{k}main.pth')
    indis_checkpoint_path = os.path.join(checkpoint_dir, f'{nside}{k}indis.pth')

    # Load checkpoints if they exist
    if os.path.exists(unet_checkpoint_path):
        inunet.load_state_dict(torch.load(unet_checkpoint_path, map_location=device))

    if os.path.exists(indis_checkpoint_path):
        indiscriminator.load_state_dict(torch.load(indis_checkpoint_path, map_location=device))

    return inunet, indiscriminator, unet_optimizer, discriminator_optimizer, writer, checkpoint_dir

def log_memory_usage():
    # Log memory usage (only if CUDA is available)
    if torch.cuda.is_available():
        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Max allocated memory: {torch.cuda.max_memory_allocated()} bytes")



def cleanup():
    dist.destroy_process_group()

    
def train_step(input_image, target, step, accumulation_steps=16, inunet=None, indiscriminator=None, unet_optimizer=None, discriminator_optimizer=None, writer=None,device=None):
    inunet.train()
    #indiscriminator.train()

    input_image = input_image.permute(0, 3, 1, 2).to(device)  # Changing from [1, 1024, 1024, 6] to [1, 6, 1024, 1024]
    target = target.unsqueeze(1).to(device)  # Changing from [1, 1024, 1024] to [1, 1, 1024, 1024]
    for name, param in inunet.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on device {param.device}, expected {device}",flush=True)

    for name, buffer in inunet.named_buffers():
        if buffer.device != device:
            print(f"Buffer {name} is on device {buffer.device}, expected {device}",flush=True)
    for name, param in indiscriminator.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on device {param.device}, expected {device}",flush=True)
    
    for name, buffer in indiscriminator.named_buffers():
        if buffer.device != device:
            print(f"Buffer {name} is on device {buffer.device}, expected {device}",flush=True)

    input_image, target = input_image.to(device), target.to(device)
        
    # Reset gradients only if it's the first step in the accumulation cycle
    if(step%accumulation_steps==0):
        unet_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
    
    # Forward pass
    geny_output = inunet(input_image)
    discy_real_output = indiscriminator(input_image, target)
    discy_generated_output = indiscriminator(input_image, geny_output.detach())
    
    # Calculate losses
    in1gen_loss, l1_loss = generator_loss(geny_output, target,discy_generated_output)
    disc_loss = discriminator_loss(discy_real_output, discy_generated_output)
    
    # Backward pass and optimization
    in1gen_loss.backward(retain_graph=True)
    disc_loss.backward()
    if((step+1)%accumulation_steps==0):
        unet_optimizer.step()
        discriminator_optimizer.step()
    
    # Log losses to TensorBoard
    writer.add_scalar('Generator/Loss', in1gen_loss.item(), step)
    writer.add_scalar('Discriminator/Loss', disc_loss.item(), step)
    writer.add_scalar('L1 Loss', l1_loss.item(), step)

    return l1_loss.item()

def train_loop(device,num_epochs, num_train_samples, num_val_samples, k, nside=1024,acc_steps=16):

    t0 = perf_counter()
    gc.collect()
    torch.cuda.empty_cache()  # Clear GPU memory
    inunet, indiscriminator, unet_optimizer, discriminator_optimizer, writer, checkpoint_dir = initialize_models(k, nside,device)

    # Move the model to the specified GPU
    inunet = inunet.to(device)
    dc=0
    indiscriminator = indiscriminator.to(device)
    
    best_l1_loss = float('inf')

    for epoch in range(num_epochs):
        for i in range(num_train_samples):
            inp, out = load_random_pair(k,nside)
            inp = inp.clone().detach().unsqueeze(0).float().to(device)
            out = out.clone().detach().unsqueeze(0).float().to(device)

            l1_loss = train_step(inp, out, epoch * num_train_samples + i, inunet=inunet, indiscriminator=indiscriminator, unet_optimizer=unet_optimizer, discriminator_optimizer=discriminator_optimizer, writer=writer,device=device)

        if epoch % 20 == 0:
            dc+=1
            torch.save(inunet.state_dict(), os.path.join(checkpoint_dir, f'{nside}{k}main.pth'))
            torch.save(indiscriminator.state_dict(), os.path.join(checkpoint_dir, f'{nside}{k}indis.pth'))    
            print(dc, flush=True)
        if l1_loss < best_l1_loss:
            best_l1_loss = l1_loss
            torch.save(inunet.state_dict(), os.path.join(checkpoint_dir, f'{nside}{k}mainbest.pth'))

        print(f"Epoch {epoch + 1}/{num_epochs} : loss {l1_loss}", flush=True)
        print(f"for k value {k} Time elapsed: {perf_counter() - t0:.2f} seconds", flush=True)
    
    writer.close()
    cleanup()







