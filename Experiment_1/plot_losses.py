import re
import matplotlib.pyplot as plt
import os
import argparse

def parse_log(log_path):
    steps = []
    loss_disc = []
    loss_gen = []
    loss_fm = []
    loss_mel = []
    loss_kl = []
    
    # regex to match lines like: INFO	[2.028, 3.152, 7.856, 8.427, 2.459, 180000, 0.00019]
    # format: [disc, gen, fm, mel, kl, step, lr]
    pattern = re.compile(r'INFO\s+\[([\d\.e-]+),\s*([\d\.e-]+),\s*([\d\.e-]+),\s*([\d\.e-]+),\s*([\d\.e-]+),\s*(\d+),\s*([\d\.e-]+)\]')

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                vals = [float(x) for x in match.groups()]
                loss_disc.append(vals[0])
                loss_gen.append(vals[1])
                loss_fm.append(vals[2])
                loss_mel.append(vals[3])
                loss_kl.append(vals[4])
                steps.append(int(vals[5]))
                
    return steps, loss_disc, loss_gen, loss_fm, loss_mel, loss_kl

def plot_losses(log_path, output_dir):
    steps, ld, lg, lfm, lmel, lkl = parse_log(log_path)
    
    if not steps:
        print("No training data found in log file.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Plot all losses
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(steps, ld, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(steps, lg, label='Generator Loss', color='orange')
    plt.title('Generator Loss')
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(steps, lfm, label='Feature Match Loss', color='green')
    plt.title('Feature Match Loss')
    plt.grid(True)
    
    plt.subplot(3, 2, 4)
    plt.plot(steps, lmel, label='Mel Loss', color='red')
    plt.title('Mel Loss')
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(steps, lkl, label='KL Loss', color='purple')
    plt.title('KL Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'training_losses.png')}")
    
    # Plot Gen vs Disc
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ld, label='Disc Loss')
    plt.plot(steps, lg, label='Gen Loss')
    plt.title('GAN Conflict (Gen vs Disc)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'gan_losses.png'))
    print(f"Plot saved to {os.path.join(output_dir, 'gan_losses.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='train.log', help='Path to train.log')
    parser.add_argument('--out', type=str, default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    plot_losses(args.log, args.out)
