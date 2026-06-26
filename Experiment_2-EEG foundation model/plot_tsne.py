import os
import glob
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from EEGModule import EEGModule
import utils

def get_eeg(filename, channels_select='all'):
    eeg = np.load(filename)
    eeg = torch.FloatTensor(eeg)
    
    # Matching data_utils.py exactly
    if channels_select == 'all':
        eeg = eeg[:128]
    else:
        eeg = eeg[channels_select]
        
    eeg = torch.nn.functional.normalize(eeg, p=2, dim=1) # time-wise normalization
    return eeg

def run(args):
    # Load hyperparameters
    hps = utils.get_hparams_from_file(args.config)
    print(f"Loaded config from {args.config}")

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
    print(f"Loading checkpoint from {args.checkpoint}")
    utils.load_checkpoint(args.checkpoint, eeg_module, None)
    eeg_module.eval()

    # Search for files
    # User's path: /media/hdd1/amkapoor/N400/N400Stimset_manuscriptdata/N400Stimset_manuscriptdata/N400_epoched/sub-02/sub-02-_-NPC_bath.wav.npy
    # Based on user feedback, the folder name is likely 'N400_epoched'
    epoched_dir_candidates = [
        os.path.join(hps.data.data_root_dir, "N400_epoched"),
        os.path.join(hps.data.data_root_dir, "2022N400_Epoched"),
        hps.data.data_root_dir
    ]
    
    epoched_dir = None
    for cand in epoched_dir_candidates:
        if os.path.exists(cand):
            # Check if it contains sub-* folders
            if glob.glob(os.path.join(cand, "sub-*")):
                epoched_dir = cand
                break
    
    if epoched_dir is None:
        print(f"Error: Could not find epoched data directory (containing sub-* folders) in {hps.data.data_root_dir}")
        print(f"Tried: {epoched_dir_candidates}")
        return

    print(f"Using epoched directory: {epoched_dir}")

    search_pattern = os.path.join(epoched_dir, "sub-*", f"sub-*-_-{args.audio_name}.npy")
    files = glob.glob(search_pattern)
    
    if not files:
        # Try without the .wav in the middle if the user provided it but the files don't have it, 
        # or vice versa. The user said: sub-02-_-NPC_bath.wav.npy
        print(f"No files found for pattern: {search_pattern}")
        return

    print(f"Found {len(files)} participant files for audio: {args.audio_name}")

    embeddings = []
    subject_ids = []

    with torch.no_grad():
        for f in files:
            sub_id = os.path.basename(os.path.dirname(f))
            print(f"Processing {sub_id}...")
            
            try:
                # Load and preprocess
                eeg = get_eeg(f, hps.data.eeg_channels if hasattr(hps.data, 'eeg_channels') else 'all')
                eeg = eeg.unsqueeze(0).to(device) # Add batch dimension: (1, Channels, Time)
                
                # Forward pass
                # EEGModule.forward returns: x, mask, mid_output, decoder_out
                _, _, mid_output, _ = eeg_module(eeg)
                
                # mid_output shape: (1, Hidden_Channels, Seq_Len)
                # Global Average Pooling over time
                embedding = mid_output.mean(dim=2).squeeze(0).cpu().numpy()
                
                embeddings.append(embedding)
                subject_ids.append(sub_id)
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if len(embeddings) < 2:
        print("Not enough embeddings to plot T-SNE.")
        return

    embeddings = np.array(embeddings)
    
    # Run T-SNE
    print("Running T-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    for i, sub_id in enumerate(subject_ids):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=sub_id)
        plt.annotate(sub_id, (embeddings_2d[i, 0], embeddings_2d[i, 1]), alpha=0.7)

    plt.title(f"T-SNE of EEG Embeddings for Audio: {args.audio_name}")
    plt.xlabel("T-SNE dimension 1")
    plt.ylabel("T-SNE dimension 2")
    # If many subjects, legend might be too large
    if len(subject_ids) < 30:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_path = f"tsne_{args.audio_name.replace('.', '_')}.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot T-SNE of EEG embeddings for a specific audio across participants.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EEG module checkpoint (E_*.pth)")
    parser.add_argument("--audio_name", type=str, required=True, help="Audio file name (e.g., NPC_bath.wav)")
    
    args = parser.parse_args()
    run(args)
