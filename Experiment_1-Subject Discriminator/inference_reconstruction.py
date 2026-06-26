import os
import argparse
import torch
import librosa
import numpy as np
from scipy.io.wavfile import write
from scipy.fftpack import dct

import utils
import commons
from models import SpeechEncoder, Generator
from mel_processing import spectrogram_torch, mel_spectrogram_torch

import logging
import warnings

# 1. Suppress Numba and Matplotlib logs
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def calculate_mcd_from_mel(gt_mel, pred_mel, n_mcc=13):
    if torch.is_tensor(gt_mel):
        gt_mel = gt_mel.cpu().numpy()
    if torch.is_tensor(pred_mel):
        pred_mel = pred_mel.cpu().numpy()
    
    if gt_mel.ndim == 3:
        gt_mel = gt_mel.squeeze(0)
    if pred_mel.ndim == 3:
        pred_mel = pred_mel.squeeze(0)
    
    if np.any(np.isnan(gt_mel)) or np.any(np.isinf(gt_mel)):
        return float('inf')
    if np.any(np.isnan(pred_mel)) or np.any(np.isinf(pred_mel)):
        return float('inf')
    
    # Transpose to [time, n_mel] for DCT computation
    gt_mel_T = gt_mel.T
    pred_mel_T = pred_mel.T
    
    # Apply DCT-II to get cepstral coefficients, keep first n_mcc (excluding 0th)
    gt_mcc = dct(gt_mel_T, type=2, axis=1, norm='ortho')[:, 1:n_mcc]
    pred_mcc = dct(pred_mel_T, type=2, axis=1, norm='ortho')[:, 1:n_mcc]
    
    # Frame-by-frame Euclidean distance
    gt_mcc_T = gt_mcc.T
    pred_mcc_T = pred_mcc.T
    
    diff = gt_mcc_T - pred_mcc_T
    dist = np.sqrt(np.sum(diff**2, axis=0))
    mcd = np.mean(dist)
    
    return mcd


def calculate_mel_corr(gt_mel, pred_mel):
    """
    Calculates Pearson correlation between two mel-spectrograms.
    Returns correlation as percentage.
    """
    if torch.is_tensor(gt_mel):
        gt_mel = gt_mel.cpu().numpy()
    if torch.is_tensor(pred_mel):
        pred_mel = pred_mel.cpu().numpy()
    
    gt_flat = gt_mel.flatten()
    pred_flat = pred_mel.flatten()
    
    if np.any(np.isnan(gt_flat)) or np.any(np.isnan(pred_flat)):
        return 0.0
    
    corr = np.corrcoef(gt_flat, pred_flat)[0, 1]
    return corr * 100  # Report as percentage

class ReconstructionModel(torch.nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.enc_q = SpeechEncoder(
            hps.data.filter_length // 2 + 1,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            5, 1, 16
        )
        self.dec = Generator(
            hps.model.inter_channels,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes
        )

    def infer(self, spec, spec_lengths):
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths)
        o = self.dec(z * y_mask)
        return o, y_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_wav', type=str, required=True, help='Path to input wav file')
    parser.add_argument('-o', '--output_wav', type=str, required=True, help='Path to save reconstructed wav file')
    parser.add_argument('-c', '--config', type=str, default="config_example.json", help='JSON file for configuration')
    parser.add_argument('-checkpoint', '--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_g = ReconstructionModel(hps).to(device)
    _ = utils.load_checkpoint(args.checkpoint_path, net_g, None)
    net_g.eval()

    # Load and process audio
    audio, sr = librosa.load(args.input_wav, sr=hps.data.sampling_rate)
    audio_norm = torch.FloatTensor(audio).unsqueeze(0).to(device)
    
    # Generate spectrogram
    spec = spectrogram_torch(
        audio_norm, 
        hps.data.filter_length,
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length,
        center=False
    )
    spec_lengths = torch.LongTensor([spec.size(-1)]).to(device)

    # Inference
    with torch.no_grad():
        y_hat, y_mask = net_g.infer(spec, spec_lengths)
        audio_out = y_hat[0].cpu().numpy()
        
        # Generate mel-spectrograms for metrics calculation
        # Input mel-spectrogram
        gt_mel = mel_spectrogram_torch(
            audio_norm,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        
        # Reconstructed mel-spectrogram
        pred_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        
        # Align mel lengths
        min_len_mel = min(gt_mel.size(2), pred_mel.size(2))
        gt_mel = gt_mel[:, :, :min_len_mel]
        pred_mel = pred_mel[:, :, :min_len_mel]
        
        # Calculate metrics
        mcd = calculate_mcd_from_mel(gt_mel, pred_mel)
        mel_corr = calculate_mel_corr(gt_mel, pred_mel)
        
    # Save audio
    write(args.output_wav, hps.data.sampling_rate, audio_out.T)
    print(f"Reconstructed audio saved to {args.output_wav}")
    print(f"\n--- Reconstruction Metrics ---")
    print(f"MCD (Mel-Cepstral Distortion): {mcd:.4f} dB")
    print(f"Mel Correlation: {mel_corr:.2f}%")

if __name__ == "__main__":
    main()
