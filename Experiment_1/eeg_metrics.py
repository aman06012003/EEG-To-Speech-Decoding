import os
import torch
import numpy as np
import argparse
from scipy import signal, stats
from EEGModule import EEGModule
import utils

def get_eeg(filename, channels_select='all', orig_fs=512, target_fs=256, in_channels=128):
    eeg = np.load(filename)
    
    # Resample from orig_fs to target_fs
    if orig_fs != target_fs:
        if eeg.shape[0] == in_channels or eeg.shape[0] == 128:
            resample_axis = 1
            num_samples = eeg.shape[1]
        elif eeg.shape[1] == in_channels or eeg.shape[1] == 128:
            resample_axis = 0
            num_samples = eeg.shape[0]
        else:
            resample_axis = 0 if eeg.shape[0] > eeg.shape[1] else 1
            num_samples = eeg.shape[resample_axis]
            
        new_num_samples = int(num_samples * target_fs / orig_fs)
        eeg = signal.resample(eeg, new_num_samples, axis=resample_axis)

    eeg = torch.FloatTensor(eeg)
    
    # Ensure (Channels, Time) orientation
    if eeg.shape[0] > 500 and eeg.shape[1] <= 136:
        eeg = eeg.transpose(0, 1)
        
    # Matching data_utils.py exactly
    if channels_select == 'all':
        eeg = eeg[:in_channels]
    else:
        eeg = eeg[channels_select]
        
    eeg = torch.nn.functional.normalize(eeg, p=2, dim=1) # time-wise normalization
    return eeg

def calculate_metrics(orig, recon, fs=256):
    """
    orig, recon: (Channels, Time) numpy arrays
    """
    num_channels = orig.shape[0]
    
    rmse_list = []
    mae_list = []
    pcc_list = []
    cosine_list = []
    psd_corr_list = []
    snr_list = []
    
    for i in range(num_channels):
        o = orig[i]
        r = recon[i]
        
        # Time Domain
        rmse = np.sqrt(np.mean((o - r)**2))
        mae = np.mean(np.abs(o - r))
        rmse_list.append(rmse)
        mae_list.append(mae)
        
        # Pearson Correlation
        if np.std(o) == 0 or np.std(r) == 0:
            pcc = 0
        else:
            pcc, _ = stats.pearsonr(o, r)
        pcc_list.append(pcc)
        
        # Cosine Similarity
        cos_sim = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r))
        cosine_list.append(cos_sim)
        
        # SNR
        noise_pow = np.mean((o - r)**2)
        if noise_pow == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(np.mean(o**2) / noise_pow)
        snr_list.append(snr)
        
        # PSD Correlation
        f_o, p_o = signal.welch(o, fs=fs, nperseg=min(len(o), 256))
        f_r, p_r = signal.welch(r, fs=fs, nperseg=min(len(r), 256))
        if np.std(p_o) == 0 or np.std(p_r) == 0:
            psd_corr = 0
        else:
            psd_corr, _ = stats.pearsonr(p_o, p_r)
        psd_corr_list.append(psd_corr)

    return {
        "RMSE": np.mean(rmse_list),
        "MAE": np.mean(mae_list),
        "PCC": np.mean(pcc_list),
        "Cosine_Sim": np.mean(cosine_list),
        "PSD_Corr": np.mean(psd_corr_list),
        "SNR": np.mean(snr_list)
    }

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

    # Calculate Metrics
    metrics = calculate_metrics(eeg_orig_np, eeg_recon_np, fs=args.target_fs)

    print("\n" + "="*40)
    print(f"RECONSTRUCTION METRICS")
    print(f"File: {os.path.basename(args.eeg_path)}")
    print("-" * 40)
    for m, val in metrics.items():
        print(f"{m:12s}: {val:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics for EEG reconstruction.")
    parser.add_argument("--config", type=str, required=True, help="Path to configs.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to EEG module checkpoint (E_*.pth)")
    parser.add_argument("--eeg_path", type=str, required=True, help="Path to the .npy EEG file")
    parser.add_argument("--orig_fs", type=int, default=512, help="Original sampling frequency in npy (default 512)")
    parser.add_argument("--target_fs", type=int, default=256, help="Target sampling frequency for model (default 256)")
    
    args = parser.parse_args()
    run(args)
