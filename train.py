from functions import train_loop
num_epochs = 40
num_train_samples = 16
num_val_samples = 4
nside = 1024

# Start training
train_loop(num_epochs, num_train_samples, num_val_samples, nside)