
import argparse
import torch
import torch
import multiprocessing
import argparse
from train import train_loop
def main():
    
    parser.add_argument('--num_epochs', type=int, default=440000, help='Number of epochs')
    parser.add_argument('--num_train_samples', type=int, default=32, help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=100, help='Number of validation samples')
    parser.add_argument('--nside', type=int, default=1024, help='accumulation steps')
    parser.add_argument('--acc_steps', type=int, default=4, help='accumulation steps')

    args = parser.parse_args()

    

    # Run training loop
    train_loop(device,num_epochs=args.num_epochs, num_train_samples=args.num_train_samples, num_val_samples=args.num_val_samples, k=[0,1,2,3,4,5,6,7,8,11,12], nside=args.nside,acc_steps=args.acc_steps)




def run_training(k, args):
    device = torch.device('cpu')  # Use CPU for all processes
    print(f"Starting process for k={k}")
    train_loop(
        device=device,
        num_epochs=args.num_epochs,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        k=k,
        nside=args.nside,
        acc_steps=args.acc_steps
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train models with different k values.')
    parser.add_argument('--k', type=int, help='Value of k')
    parser.add_argument('--gpu', type=int, help='GPU device ID')
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--num_train_samples", type=int, default=32)
    parser.add_argument("--num_val_samples", type=int, default=10)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--acc_steps", type=int, default=8)
    args = parser.parse_args()
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: cuda:{args.gpu}")
        train_loop(device=device,num_epochs=args.num_epochs,num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,k=args.k,nside=args.nside,acc_steps=args.acc_steps)
    else:
        device = torch.device('cpu')
        print("Using CPU")
        k_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Start multiple processes for each k value
        processes = []
        for k in k_values:
            p = multiprocessing.Process(target=run_training, args=(k, args))
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        print("All processes completed.")


