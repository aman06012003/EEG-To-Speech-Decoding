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
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found at {args.config}")
        return
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
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        print("Note: If using absolute paths on Linux, ensure it starts with a forward slash (/).")
        return
        
    print(f"Loading checkpoint from {args.checkpoint}")
    utils.load_checkpoint(args.checkpoint, eeg_module, None)
    eeg_module.eval()

    # Search for files
    epoched_dir_candidates = [
        os.path.join(hps.data.data_root_dir, "N400_epoched"),
        os.path.join(hps.data.data_root_dir, "2022N400_Epoched"),
        hps.data.data_root_dir
    ]
    
    epoched_dir = None
    for cand in epoched_dir_candidates:
        if os.path.exists(os.path.join(cand, args.subject_id)):
            epoched_dir = cand
            break
    
    if epoched_dir is None:
        print(f"Error: Could not find subject folder {args.subject_id} in any of {epoched_dir_candidates}")
        return

    subject_folder = os.path.join(epoched_dir, args.subject_id)
    print(f"Using subject folder: {subject_folder}")

    # Find all .npy files for this subject
    files = glob.glob(os.path.join(subject_folder, "*.npy"))
    
    if not files:
        print(f"No .npy files found in: {subject_folder}")
        return

    print(f"Found {len(files)} EEG files for subject: {args.subject_id}")

    embeddings = []
    audio_names = []

    with torch.no_grad():
        for f in files:
            # Extract audio name from filename: sub-03-_-NPC_bath.wav.npy -> NPC_bath.wav
            filename = os.path.basename(f)
            try:
                # The pattern is sub-XX-_-AUDIO_NAME.npy
                audio_name = filename.split('-_-')[1].replace('.npy', '')
            except IndexError:
                audio_name = filename # fallback
            
            print(f"Processing {audio_name}...")
            
            try:
                # Load and preprocess
                eeg = get_eeg(f, hps.data.eeg_channels if hasattr(hps.data, 'eeg_channels') else 'all')
                eeg = eeg.unsqueeze(0).to(device) # Add batch dimension: (1, Channels, Time)
                
                # Forward pass
                _, _, mid_output, _ = eeg_module(eeg)
                
                # Global Average Pooling over time
                embedding = mid_output.mean(dim=2).squeeze(0).cpu().numpy()
                
                embeddings.append(embedding)
                audio_names.append(audio_name)
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
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=range(len(audio_names)), cmap='viridis', alpha=0.6)
    
    if len(audio_names) < 50:
        for i, audio_name in enumerate(audio_names):
            plt.annotate(audio_name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.8)
    else:
        for i in range(0, len(audio_names), max(1, len(audio_names)//20)):
             plt.annotate(audio_names[i], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.8)

    plt.title(f"T-SNE of EEG Embeddings for Subject: {args.subject_id}")
    plt.xlabel("T-SNE dimension 1")
    plt.ylabel("T-SNE dimension 2")
    
    plt.tight_layout()
    output_path = f"tsne_subject_{args.subject_id}.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot T-SNE of EEG embeddings for all audios of a specific subject.")
    parser.add_argument("--config", type=str, required=True, help="Path to configs.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EEG module checkpoint (E_*.pth)")
    parser.add_argument("--subject_id", type=str, required=True, help="Subject ID (e.g., sub-03)")
    
    args = parser.parse_args()
    run(args)
