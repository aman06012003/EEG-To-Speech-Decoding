import argparse
import torch
import numpy as np
import librosa
import os
import utils
from mel_processing import mel_spectrogram_torch

def calculate_mcd(gt_audio, pred_audio, sr=22050, n_mfcc=80, hop_length=256, win_length=1024):
    """
    Calculates Mel-Cepstral Distortion (MCD) between ground truth and predicted audio.
    MCD = (10 * sqrt(2) / ln(10)) * mean(sqrt(sum((MCC_gt - MCC_pred)^2)))
    """
    # Ensure inputs are numpy arrays
    if torch.is_tensor(gt_audio):
        gt_audio = gt_audio.cpu().numpy()
    if torch.is_tensor(pred_audio):
        pred_audio = pred_audio.cpu().numpy()
        
    # Squeeze if necessary (handle batch dim 1)
    if gt_audio.ndim > 1:
        gt_audio = gt_audio.squeeze()
    if pred_audio.ndim > 1:
        pred_audio = pred_audio.squeeze()

    # Calculate MFCCs
    gt_mfcc = librosa.feature.mfcc(y=gt_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, center=False)
    pred_mfcc = librosa.feature.mfcc(y=pred_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, center=False)

    # Truncate to min length
    min_len = min(gt_mfcc.shape[1], pred_mfcc.shape[1])
    gt_mfcc = gt_mfcc[:, :min_len]
    pred_mfcc = pred_mfcc[:, :min_len]
    
    # Calculate squared difference
    diff = gt_mfcc - pred_mfcc
    dist = np.sum(diff**2, axis=0)
    dist = np.sqrt(dist)
    
    # Mean over frames
    mean_dist = np.mean(dist)
    
    # Scale factor
    k = 10 * np.sqrt(2) / np.log(10)
    
    mcd = k * mean_dist
    return mcd

def calculate_mel_corr(gt_mel, pred_mel):
    """
    Calculates Pearson correlation between two mel-spectrograms.
    """
    if torch.is_tensor(gt_mel):
        gt_mel = gt_mel.cpu().numpy()
    if torch.is_tensor(pred_mel):
        pred_mel = pred_mel.cpu().numpy()
        
    # Flatten to 1D arrays for correlation
    gt_flat = gt_mel.flatten()
    pred_flat = pred_mel.flatten()
    
    # Check for NaN or Inf
    if np.any(np.isnan(gt_flat)) or np.any(np.isnan(pred_flat)):
        return 0.0
        
    corr = np.corrcoef(gt_flat, pred_flat)[0, 1]
    return corr * 100 # Report as percentage

def main():
    parser = argparse.ArgumentParser(description="Calculate MCD and Mel-Corr between two audio files.")
    parser.add_argument('--original', type=str, required=True, help='Path to the original (ground truth) audio file')
    parser.add_argument('--generated', type=str, required=True, help='Path to the generated audio file')
    parser.add_argument('--config', type=str, required=True, help='Path to config.json for audio parameters')
    args = parser.parse_args()

    # Load config
    hps = utils.get_hparams_from_file(args.config)
    
    # Load audio
    print(f"Loading audio files...")
    y_gt, sr_gt = librosa.load(args.original, sr=hps.data.sampling_rate)
    y_gen, sr_gen = librosa.load(args.generated, sr=hps.data.sampling_rate)
    
    # Ensure same length for convenience (though metrics handle truncation)
    min_len = min(len(y_gt), len(y_gen))
    y_gt = y_gt[:min_len]
    y_gen = y_gen[:min_len]
    
    # Calculate MCD
    print("Calculating MCD...")
    mcd = calculate_mcd(y_gt, y_gen, sr=hps.data.sampling_rate)
    
    # Calculate Mel-Corr
    print("Calculating Mel-Corr...")
    # Convert to torch for mel calculation
    y_gt_torch = torch.FloatTensor(y_gt).unsqueeze(0)
    y_gen_torch = torch.FloatTensor(y_gen).unsqueeze(0)
    
    y_gt_mel = mel_spectrogram_torch(
        y_gt_torch, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
    )
    
    y_gen_mel = mel_spectrogram_torch(
        y_gen_torch, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
    )
    
    mel_corr = calculate_mel_corr(y_gt_mel, y_gen_mel)
    
    print("-" * 30)
    print(f"Results:")
    print(f"MCD: {mcd:.4f} dB")
    print(f"Mel-Corr: {mel_corr:.4f} %")
    print("-" * 30)

if __name__ == "__main__":
    main()
