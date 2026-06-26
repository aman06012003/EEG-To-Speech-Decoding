import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

def run(args):
    if not os.path.isfile(args.eeg_path):
        print(f"Error: EEG file not found at {args.eeg_path}")
        return
    
    print(f"Loading EEG from {args.eeg_path}")
    eeg = np.load(args.eeg_path)
    # Check shape. User says 128 channels. 
    # Usually files are (Time, Channels) or (Channels, Time).
    # If 128 is one of the dims, we can identify it.
    if eeg.shape[0] != 128 and eeg.shape[1] == 128:
        eeg = eeg.T
        
    num_channels, num_samples = eeg.shape
    fs = args.fs
    time = np.arange(num_samples) / fs

    # Channel selection
    if args.channels:
        ch_indices = args.channels
    else:
        # Default to first 16 or something reasonable for a preview
        ch_indices = list(range(min(16, num_channels)))

    # Vertical offset for stacking
    # We'll calculate a scale for offset based on the std of the signals
    scale = np.max(np.std(eeg[ch_indices], axis=1)) * 5 if args.auto_scale else args.offset

    plt.figure(figsize=(15, 10))
    
    for i, ch_idx in enumerate(ch_indices):
        # We plot from bottom to top for more natural indexing or top to bottom?
        # Let's do top to bottom (first channel at the top)
        offset = -i * scale
        plt.plot(time, eeg[ch_idx] + offset, color='black', linewidth=0.5)
        plt.text(-0.01 * time[-1], offset, f'Ch {ch_idx}', verticalalignment='center', horizontalalignment='right', fontsize=8)

    plt.title(f'EEG Signal: {os.path.basename(args.eeg_path)} ({len(ch_indices)} channels shown)')
    plt.xlabel('Time (seconds)')
    plt.yticks([]) # Hide Y ticks as they are relative to offsets
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # If plotting 128 channels, it might still be crowded, but the offset helps.
    if len(ch_indices) > 20:
        plt.gcf().set_size_inches(15, 0.5 * len(ch_indices))

    plt.tight_layout()
    output_path = f"plot_{os.path.basename(args.eeg_path).replace('.npy', '')}.png"
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correctly plot EEG signal from .npy file.")
    parser.add_argument("--eeg_path", type=str, required=True, help="Path to the .npy EEG file")
    parser.add_argument("--fs", type=int, default=512, help="Sampling frequency (default 256)")
    parser.add_argument("--channels", type=int, nargs='+', help="Specific channel indices to plot (e.g. 0 1 2 ...)")
    parser.add_argument("--offset", type=float, default=50.0, help="Vertical offset between channels (standard units)")
    parser.add_argument("--auto_scale", action="store_true", help="Automatically calculate offset based on signal variance")
    
    args = parser.parse_args()
    run(args)
