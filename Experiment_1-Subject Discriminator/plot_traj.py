import os
import torch
import numpy as np
import argparse
from scipy import signal
import matplotlib.pyplot as plt
from EEGModule import EEGModule
import utils

def get_eeg(filename, channels_select='all', orig_fs=512, target_fs=256, in_channels=128):
    eeg = np.load(filename)
    
    # Resample from orig_fs to target_fs
    if orig_fs != target_fs:
        # Determine which axis is time. Usually the larger one or the one not matching in_channels.
        if eeg.shape[0] == in_channels or eeg.shape[0] == 128:
            resample_axis = 1
            num_samples = eeg.shape[1]
        elif eeg.shape[1] == in_channels or eeg.shape[1] == 128:
            resample_axis = 0
            num_samples = eeg.shape[0]
        else:
            # Fallback to larger dimension
            resample_axis = 0 if eeg.shape[0] > eeg.shape[1] else 1
            num_samples = eeg.shape[resample_axis]
            
        new_num_samples = int(num_samples * target_fs / orig_fs)
        eeg = signal.resample(eeg, new_num_samples, axis=resample_axis)

    eeg = torch.FloatTensor(eeg)
    
    # Ensure (Channels, Time) orientation
    # If the first dimension is very large (likely time) and second is small (likely channels), transpose.
    if eeg.shape[0] > 500 and eeg.shape[1] <= 136:
        eeg = eeg.transpose(0, 1)
        
    # Matching data_utils.py exactly
    if channels_select == 'all':
        eeg = eeg[:in_channels]
    else:
        eeg = eeg[channels_select]
        
    eeg = torch.nn.functional.normalize(eeg, p=2, dim=1) # time-wise normalization
    return eeg

def run(args):
    # Load hyperparameters
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
    hps = utils.get_hparams_from_file(args.config)

    # Initialize EEGModule
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eeg_module = EEGModule(
        n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
        use_s4=hps.model.eeg_module.use_s4,
        n_layers_s4=hps.model.eeg_module.n_layers_s4,
        embedding_size=hps.model.inter_channels,
        is_mask=False,
        in_channels=hps.model.eeg_module.in_channels,
        device=device
    ).to(device)

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    utils.load_checkpoint(args.checkpoint, eeg_module, None)
    eeg_module.eval()

    # Load and reconstruct
    channels_select = hps.data.eeg_channels if hasattr(hps.data, 'eeg_channels') else 'all'
    eeg_orig = get_eeg(args.eeg_path, channels_select, orig_fs=args.orig_fs, target_fs=args.target_fs, in_channels=hps.model.eeg_module.in_channels)
    eeg_input = eeg_orig.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, eeg_recon = eeg_module(eeg_input)
        eeg_recon_np = eeg_recon.squeeze(0).cpu().numpy()
        eeg_orig_np = eeg_orig.cpu().numpy()

    num_channels, num_samples = eeg_recon_np.shape
    fs = args.target_fs
    time = np.arange(num_samples) / fs

    # Channel selection
    ch_indices = args.channels if args.channels else list(range(min(16, num_channels)))
    
    # Scaling for stacked plot
    if args.auto_scale:
        scale = np.max(np.std(eeg_recon_np[ch_indices], axis=1)) * 5
    else:
        scale = args.offset

    plt.figure(figsize=(15, 0.6 * len(ch_indices) if len(ch_indices) > 10 else 10))
    
    for i, ch_idx in enumerate(ch_indices):
        offset = -i * scale
        
        # Plot original (optional/faded) or just reconstructed
        if args.show_orig:
            plt.plot(time, eeg_orig_np[ch_idx] + offset, color='blue', linewidth=0.5, alpha=0.3, label='Original' if i==0 else "")
            
        plt.plot(time, eeg_recon_np[ch_idx] + offset, color='red', linewidth=0.7, alpha=0.9, label='Reconstructed' if i==0 else "")
        
        plt.text(-0.01 * time[-1], offset, f'Ch {ch_idx}', verticalalignment='center', horizontalalignment='right', fontsize=9)

    plt.title(f'Stacked Reconstructed EEG Signal: {os.path.basename(args.eeg_path)}\n(Red = Reconstructed' + (' , Blue = Original' if args.show_orig else '') + ')')
    plt.xlabel('Time (seconds)')
    plt.yticks([])
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    if args.show_orig:
        plt.legend(loc='upper right')

    plt.tight_layout()
    output_path = f"recon_stacked_{os.path.basename(args.eeg_path).replace('.npy', '')}.png"
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot stacked reconstructed EEG signals.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EEG module checkpoint (E_*.pth)")
    parser.add_argument("--eeg_path", type=str, required=True, help="Path to the original .npy EEG file")
    parser.add_argument("--orig_fs", type=int, default=512, help="Original sampling frequency in npy (default 512)")
    parser.add_argument("--target_fs", type=int, default=256, help="Target sampling frequency for model (default 256)")
    parser.add_argument("--channels", type=int, nargs='+', help="Specific channel indices to plot")
    parser.add_argument("--offset", type=float, default=2.0, help="Vertical offset between channels")
    parser.add_argument("--auto_scale", action="store_true", help="Auto-calculate offset")
    parser.add_argument("--show_orig", action="store_true", help="Overlay original signal in faint blue")
    
    args = parser.parse_args()
    run(args)
