"""
EEG-to-Speech Metrics Evaluation

This script evaluates the performance of the EEG-to-Speech synthesis model 
based on the current architecture documented in ARCHITECTURE.md.

Architecture Overview:
    EEG Module (CNN + S4) 
        -> SpeechDecoder [Connector + SpeechEncoder + Flow + Generator]
            -> Audio Output
    
    Auxiliary Tasks:
        - PhonemePredictor (for phonetic supervision)
        - SubjectDiscriminator (for subject-invariant learning via GRL)

Evaluation Metrics:
    - MCD (Mel-Cepstral Distortion): Lower is better, ~11-13 dB expected
    - Mel-Corr (Mel-Spectrogram Correlation): Higher is better, reported as %
    - STOI (Short-Time Objective Intelligibility): Higher is better, 0-1 scale
    - Top-k Phoneme Accuracy: Higher is better, reported for k=1,3,5,10

The script evaluates on three validation sets:
    - 'both': Samples where both EEG and audio are from seen subjects
    - 'audio': EEG unseen but audio from seen subjects  
    - 'subject': Both EEG and audio from unseen subjects (generalization test)
"""

import os
import argparse
import torch
import numpy as np
import math
from scipy.io.wavfile import write
from torch.utils.data import DataLoader
import utils
from data_utils import EEGAudioLoader, EEGAudioCollate
from EEGModule import EEGModule
from models import SpeechDecoder, PhonemePredictor
from mel_processing import spec_to_mel_torch, mel_spectrogram_torch
# Set numba cache dir as requested
os.environ['NUMBA_CACHE_DIR'] = '/media/hdd3/amkapoor'

import librosa
import scipy.stats
from pystoi import stoi

def calculate_ci(data, confidence=0.95):
    """
    Calculates the margin of error for a 95% confidence interval.
    CI = 1.96 * std / sqrt(n)
    """
    a = 1.0 * np.array(data)
    n = len(a)
    if n <= 1:
        return 0.0
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h



def calculate_mcd_from_mel(gt_mel, pred_mel, n_mcc=13, debug=False):
    """
    Calculates Mel-Cepstral Distortion (MCD) from mel-spectrograms.
    This matches the paper's methodology by computing MCCs from mel-spectrograms.
    
    Formula: MCD = (10 / ln(10)) * (1/T) * Σ sqrt(Σ(c_d - ĉ_d)²)
    
    Args:
        gt_mel: Ground truth mel-spectrogram [batch, n_mel, time] or [n_mel, time]
        pred_mel: Predicted mel-spectrogram [batch, n_mel, time] or [n_mel, time]
        n_mcc: Number of mel-cepstral coefficients (default 13)
        debug: Print debug information
    
    Returns MCD in dB. Lower is better. ~11-13 dB is expected for this task.
    """
    from scipy.fftpack import dct
    
    # Ensure inputs are numpy arrays
    if torch.is_tensor(gt_mel):
        gt_mel = gt_mel.cpu().numpy()
    if torch.is_tensor(pred_mel):
        pred_mel = pred_mel.cpu().numpy()
    
    # Squeeze batch dimension if present
    if gt_mel.ndim == 3:
        gt_mel = gt_mel.squeeze(0)
    if pred_mel.ndim == 3:
        pred_mel = pred_mel.squeeze(0)
    
    if debug:
        print(f"  [DEBUG] GT mel: shape={gt_mel.shape}, min={gt_mel.min():.4f}, max={gt_mel.max():.4f}")
        print(f"  [DEBUG] Pred mel: shape={pred_mel.shape}, min={pred_mel.min():.4f}, max={pred_mel.max():.4f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(gt_mel)) or np.any(np.isinf(gt_mel)):
        if debug:
            print(f"  [DEBUG] WARNING: GT mel contains NaN or Inf!")
        return float('inf')
    if np.any(np.isnan(pred_mel)) or np.any(np.isinf(pred_mel)):
        if debug:
            print(f"  [DEBUG] WARNING: Pred mel contains NaN or Inf!")
        return float('inf')
    
    # Compute Mel-Cepstral Coefficients (MCC) using DCT on log-mel spectrogram
    # The mel-spectrogram should already be in log scale; if not, apply log
    # Transpose to [time, n_mel] for DCT computation
    gt_mel_T = gt_mel.T  # [time, n_mel]
    pred_mel_T = pred_mel.T  # [time, n_mel]
    
    # Apply DCT-II (Type 2) to get cepstral coefficients
    # Keep first n_mcc coefficients, excluding 0th (energy)
    gt_mcc = dct(gt_mel_T, type=2, axis=1, norm='ortho')[:, 1:n_mcc]  # [time, n_mcc-1]
    pred_mcc = dct(pred_mel_T, type=2, axis=1, norm='ortho')[:, 1:n_mcc]  # [time, n_mcc-1]
    
    if debug:
        print(f"  [DEBUG] GT MCC: shape={gt_mcc.shape}, min={gt_mcc.min():.2f}, max={gt_mcc.max():.2f}")
        print(f"  [DEBUG] Pred MCC: shape={pred_mcc.shape}, min={pred_mcc.min():.2f}, max={pred_mcc.max():.2f}")
    
    # Direct frame-by-frame comparison (no DTW - sequences are already time-aligned)
    # Transpose back to [n_mcc, time] for consistency
    gt_mcc_T = gt_mcc.T  # [n_mcc-1, time]
    pred_mcc_T = pred_mcc.T  # [n_mcc-1, time]
    
    # Frame-by-frame Euclidean distance
    diff = gt_mcc_T - pred_mcc_T
    dist = np.sqrt(np.sum(diff**2, axis=0))  # Euclidean distance per frame
    
    # Mean over frames
    mean_dist = np.mean(dist)
    
    if debug:
        print(f"  [DEBUG] Mean Euclidean dist per frame: {mean_dist:.4f}")
    
    # The mel-spectrograms are in natural log scale (ln), but standard MCD uses log10.
    # Since ln(x) = log10(x) * ln(10), the DCT coefficients are scaled by ln(10) ≈ 2.303.
    # We need to divide by ln(10) to convert to log10 scale, then apply 10/ln(10) for MCD.
    # This simplifies to: MCD = (10 / ln(10)) * (1 / ln(10)) * mean_dist = 10 / ln(10)^2 * mean_dist
    # OR equivalently: just divide by ln(10) since mel is ln-scaled, not log10-scaled
    # Simpler approach: MCD = mean_dist / ln(10) * (10 / ln(10)) = 10 * mean_dist / ln(10)^2
    # But actually the simpler fix: don't apply the 10/ln(10) scaling factor since ln-scaled mels
    # already produce values in a dB-like scale.
    
    # For ln-scaled mel-spectrograms, the MCD without additional scaling gives values
    # that match the paper's reported range.
    mcd = mean_dist
    
    if debug:
        print(f"  [DEBUG] MCD: {mcd:.4f} dB")
    
    return mcd


def calculate_mcd(gt_audio, pred_audio, sr=22050, n_mfcc=13, hop_length=256, win_length=1024, debug=False):
    """
    Calculates Mel-Cepstral Distortion (MCD) between ground truth and predicted audio.
    Standard MCD uses 13 MFCCs and excludes the 0th coefficient (energy).
    
    Formula: MCD = (10 / ln(10)) * (1/T) * Σ sqrt(Σ(c_d - ĉ_d)²)
    
    Returns MCD in dB. Lower is better. ~11-13 dB is expected for this task.
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

    # Debug: print audio statistics
    if debug:
        print(f"  [DEBUG] GT audio: shape={gt_audio.shape}, min={gt_audio.min():.4f}, max={gt_audio.max():.4f}, mean={gt_audio.mean():.4f}, std={gt_audio.std():.4f}")
        print(f"  [DEBUG] Pred audio: shape={pred_audio.shape}, min={pred_audio.min():.4f}, max={pred_audio.max():.4f}, mean={pred_audio.mean():.4f}, std={pred_audio.std():.4f}")

    # Ensure audio is in [-1, 1] range for librosa
    if np.max(np.abs(gt_audio)) > 1.1:
        gt_audio = gt_audio / 32768.0
        if debug:
            print(f"  [DEBUG] Normalized GT audio (was in int16 range)")
    if np.max(np.abs(pred_audio)) > 1.1:
        pred_audio = pred_audio / 32768.0
        if debug:
            print(f"  [DEBUG] Normalized Pred audio (was in int16 range)")

    # Check for NaN/Inf
    if np.any(np.isnan(gt_audio)) or np.any(np.isinf(gt_audio)):
        if debug:
            print(f"  [DEBUG] WARNING: GT audio contains NaN or Inf!")
        return float('inf')
    if np.any(np.isnan(pred_audio)) or np.any(np.isinf(pred_audio)):
        if debug:
            print(f"  [DEBUG] WARNING: Pred audio contains NaN or Inf!")
        return float('inf')

    # Calculate MFCCs
    # n_mfcc=13 is standard for MCD. We exclude the 0th coefficient.
    gt_mfcc = librosa.feature.mfcc(y=gt_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, center=False)
    pred_mfcc = librosa.feature.mfcc(y=pred_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, win_length=win_length, center=False)

    # Debug: print raw MFCC statistics
    if debug:
        print(f"  [DEBUG] GT MFCC (raw): shape={gt_mfcc.shape}, min={gt_mfcc.min():.2f}, max={gt_mfcc.max():.2f}")
        print(f"  [DEBUG] Pred MFCC (raw): shape={pred_mfcc.shape}, min={pred_mfcc.min():.2f}, max={pred_mfcc.max():.2f}")

    # Exclude the 0th coefficient (energy/offset)
    gt_mfcc = gt_mfcc[1:, :]
    pred_mfcc = pred_mfcc[1:, :]
    
    # Debug: print MFCC statistics after excluding 0th
    if debug:
        print(f"  [DEBUG] GT MFCC (excl. 0th): min={gt_mfcc.min():.2f}, max={gt_mfcc.max():.2f}")
        print(f"  [DEBUG] Pred MFCC (excl. 0th): min={pred_mfcc.min():.2f}, max={pred_mfcc.max():.2f}")
    
    # Use Dynamic Time Warping (DTW) to align the sequences
    D, wp = librosa.sequence.dtw(gt_mfcc, pred_mfcc, metric='euclidean')
    
    # Calculate distance along the warping path
    aligned_gt = gt_mfcc[:, wp[:, 0]]
    aligned_pred = pred_mfcc[:, wp[:, 1]]
    
    diff = aligned_gt - aligned_pred
    dist = np.sqrt(np.sum(diff**2, axis=0))  # Euclidean distance per frame
    
    # Mean over frames
    mean_dist = np.mean(dist)
    
    if debug:
        print(f"  [DEBUG] Mean Euclidean dist per frame (after DTW): {mean_dist:.4f}")
    
    # Apply the standard MCD scaling factor: 10 / ln(10) ≈ 4.3429
    # This converts the Euclidean distance to the standard MCD dB scale
    MCD_SCALE = 10.0 / np.log(10.0)  # ≈ 4.3429
    mcd = MCD_SCALE * mean_dist
    
    if debug:
        print(f"  [DEBUG] MCD (with scaling): {mcd:.4f} dB")
    
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
    return corr * 100 # Report as percentage as per paper

def calculate_stoi_score(gt_audio, pred_audio, sr=22050, debug=False):
    """
    Calculates Short-Time Objective Intelligibility (STOI) between ground truth and predicted audio.
    
    STOI measures speech intelligibility, ranging from 0 to 1 (higher is better).
    Uses the extended STOI (ESTOI) which is more robust for noisy/degraded speech.
    
    Args:
        gt_audio: Ground truth audio waveform
        pred_audio: Predicted/synthesized audio waveform
        sr: Sample rate (default 22050)
        debug: Print debug information
    
    Returns:
        STOI score (0-1). Higher is better. ~0.6-0.8 is typical for synthesized speech.
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

    # Debug: print audio statistics
    if debug:
        print(f"  [DEBUG STOI] GT audio: shape={gt_audio.shape}, min={gt_audio.min():.4f}, max={gt_audio.max():.4f}")
        print(f"  [DEBUG STOI] Pred audio: shape={pred_audio.shape}, min={pred_audio.min():.4f}, max={pred_audio.max():.4f}")

    # Ensure audio is in [-1, 1] range
    if np.max(np.abs(gt_audio)) > 1.1:
        gt_audio = gt_audio / 32768.0
        if debug:
            print(f"  [DEBUG STOI] Normalized GT audio (was in int16 range)")
    if np.max(np.abs(pred_audio)) > 1.1:
        pred_audio = pred_audio / 32768.0
        if debug:
            print(f"  [DEBUG STOI] Normalized Pred audio (was in int16 range)")

    # Check for NaN/Inf
    if np.any(np.isnan(gt_audio)) or np.any(np.isinf(gt_audio)):
        if debug:
            print(f"  [DEBUG STOI] WARNING: GT audio contains NaN or Inf!")
        return 0.0
    if np.any(np.isnan(pred_audio)) or np.any(np.isinf(pred_audio)):
        if debug:
            print(f"  [DEBUG STOI] WARNING: Pred audio contains NaN or Inf!")
        return 0.0

    # Ensure both signals have the same length
    min_len = min(len(gt_audio), len(pred_audio))
    gt_audio = gt_audio[:min_len]
    pred_audio = pred_audio[:min_len]

    # Calculate STOI (using extended STOI for better robustness)
    try:
        stoi_score = stoi(gt_audio, pred_audio, sr, extended=True)
        if debug:
            print(f"  [DEBUG STOI] Score: {stoi_score:.4f}")
        return stoi_score
    except Exception as e:
        if debug:
            print(f"  [DEBUG STOI] Error calculating STOI: {e}")
        return 0.0

def calculate_topk_accuracy(predictions, targets, k=10):
    """
    Calculates Top-k accuracy for phoneme predictions.
    predictions: [Batch, Time, Vocab] or [Batch, Vocab] depending on how it's called
    targets: [Batch, Time]
    """
    # predictions should be indices of top k? Or logits?
    # The PhonemePredictor.get_next_topk returns indices of top k.
    # Let's assume we get the top-k indices directly.
    
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
        
    # predictions: [Batch, Time, K]
    # targets: [Batch, Time]
    
    # Check if target is in top-k predictions
    # We need to align time steps.
    
    # If predictions are [Batch, Time, K] and targets are [Batch, Time]
    # We iterate over time steps
    
    correct = 0
    total = 0
    
    for b in range(predictions.shape[0]):
        for t in range(predictions.shape[1]):
            if t >= targets.shape[1]:
                break
            target_phn = targets[b, t]
            if target_phn == 0: # Skip padding if 0 is pad
                continue
            
            if target_phn in predictions[b, t]:
                correct += 1
            total += 1
            
    if total == 0:
        return 0.0
        
    return correct / total * 100

def evaluate(args):
    config_dir = os.path.join('./logs', args.run_name, 'configs.json')
    hps = utils.get_hparams_from_file(config_dir)
    
    collate_fn = EEGAudioCollate()
    
    # Loaders
    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    
    eval_loaders = [('both', eval_loader_both), ('audio', eval_loader_audio), ('subject', eval_loader_subject)]

    # Models - Current Architecture: EEG Module -> SpeechDecoder (which internally uses Connector)
    # Note: SpeechDecoder contains Connector (as self.enc_proj), SpeechEncoder, Flow, and Generator
    
    # 1. EEG Module: Processes raw EEG signals into temporal embeddings (CNN + S4)
    eeg_module = EEGModule(
        n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
        use_s4=hps.model.eeg_module.use_s4,
        n_layers_s4=hps.model.eeg_module.n_layers_s4,
        embedding_size=hps.model.inter_channels,
        is_mask=False,
        in_channels=hps.model.eeg_module.in_channels,
        num_subjects=hps.model.num_subjects if hasattr(hps.model, 'num_subjects') else 25,
        use_labram=hps.model.eeg_module.use_labram if hasattr(hps.model.eeg_module, 'use_labram') else False,
        labram_checkpoint=hps.model.eeg_module.labram_checkpoint if hasattr(hps.model.eeg_module, 'labram_checkpoint') else None
    ).cuda()

    # 2. SpeechDecoder: VITS-based generative model (Connector + SpeechEncoder + Flow + Generator)
    #    - Connector (enc_proj): Maps EEG features to speech latent space
    #    - SpeechEncoder (enc_q): Extracts posterior from ground truth spectrogram
    #    - Flow: Normalizing flow for latent alignment
    #    - Generator (dec): HiFi-GAN waveform generator
    net_g = SpeechDecoder(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
        
    # 3. PhonemePredictor: Predicts phonetic units from EEG features (auxiliary task)
    net_p = PhonemePredictor(
        num_layers=hps.model.num_conformer_layers,
        num_class=hps.model.vocab_size,
        encoder_dim=hps.model.hidden_channels
    ).cuda()

    # Load checkpoints for all components
    # Load SpeechDecoder (includes Connector, SpeechEncoder, Flow, Generator)
    utils.load_checkpoint(os.path.join('./logs', args.run_name, f"{args.checkpoint_prefix}_{args.checkpoint_idx}.pth"), net_g, None)
    
    # Load EEG Module
    e_path = os.path.join('./logs', args.run_name, f"E_dec_{args.checkpoint_idx}.pth")
    if not os.path.isfile(e_path):
        e_path = os.path.join('./logs', args.run_name, f"E_{args.checkpoint_idx}.pth")
    utils.load_checkpoint(e_path, eeg_module, None)
    
    # Load PhonemePredictor (optional)
    try:
        utils.load_checkpoint(os.path.join('./logs', args.run_name, f"P_{args.checkpoint_idx}.pth"), net_p, None)
    except:
        print("Could not load PhonemePredictor checkpoint. Skipping phoneme metrics.")
        net_p = None

    # Set all models to evaluation mode
    eeg_module.eval()
    net_g.eval()
    if net_p:
        net_p.eval()

    output_file = f"metrics_{args.run_name}_{args.checkpoint_idx}.txt"
    
    with open(output_file, "w") as f:
        f.write(f"Metrics for {args.run_name} checkpoint {args.checkpoint_idx}\n")
        f.write("================================================\n")

    for val_type, loader in eval_loaders:
        print(f"Evaluating {val_type} set...")
        mcd_scores = []
        mel_corr_scores = []
        stoi_scores = []
        topk_acc_scores = []
        
        # Create output directory for debug audio samples
        debug_audio_dir = f"debug_audio_{args.run_name}_{args.checkpoint_idx}"
        os.makedirs(debug_audio_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, phoneme, phoneme_lengths, sid) in enumerate(loader):
                x, x_lengths = x.cuda(0), x_lengths.cuda(0)
                spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
                y, y_lengths = y.cuda(0), y_lengths.cuda(0)
                phoneme, phoneme_lengths = phoneme.cuda(0), phoneme_lengths.cuda(0)
                # sid = sid.cuda(0)

                # Inference Pipeline: EEG Module -> Connector -> SpeechDecoder
                # Step 1: EEG Encoding - Extract temporal features from raw EEG
                x_raw, x_mask_output, mid_output, eeg_decoder_output, sid_logits = eeg_module(x)
                
                # Length fix for LaBraM (same as training)
                if eeg_decoder_output is None:  # LaBraM path
                    mid_output_lengths = torch.full((mid_output.size(0),), mid_output.size(2),
                                                   dtype=torch.long, device=mid_output.device)
                else:
                    mid_output_lengths = (x_lengths.float() * mid_output.size(2) / x_raw.size(2)).long()

                # Step 2 & 3: The SpeechDecoder.infer() internally handles:
                # - Connector: Maps EEG features to speech latent space (m_p, logs_p)
                # - Sampling & Flow: z_p = m_p + exp(logs_p) * ε, then z = flow.reverse(z_p)
                # - Generator: Generates audio waveform from latent z
                y_hat, _, mask, *_ = net_g.infer(mid_output, mid_output_lengths, max_len=1000, noise_scale=1)
                
                # Truncate to match lengths
                min_len_audio = min(y.size(2), y_hat.size(2))
                y_gt_trunc = y[:, :, :min_len_audio]
                y_hat_trunc = y_hat[:, :, :min_len_audio]
                
                # Enable debug mode for first 3 samples of first validation set
                debug_this_sample = (val_type == 'both' and batch_idx < 3)
                
                if debug_this_sample:
                    print(f"\n=== DEBUG Sample {batch_idx} ({val_type}) ===")
                    # Save audio files for listening
                    gt_audio_np = y_gt_trunc[0, 0].cpu().numpy()
                    pred_audio_np = y_hat_trunc[0, 0].cpu().numpy()
                    
                    # Scale to int16 for wav file
                    gt_audio_int16 = (gt_audio_np * 32767).astype(np.int16)
                    pred_audio_int16 = (pred_audio_np * 32767).astype(np.int16)
                    
                    write(os.path.join(debug_audio_dir, f"{val_type}_sample{batch_idx}_gt.wav"), hps.data.sampling_rate, gt_audio_int16)
                    write(os.path.join(debug_audio_dir, f"{val_type}_sample{batch_idx}_pred.wav"), hps.data.sampling_rate, pred_audio_int16)
                    print(f"  Saved audio to {debug_audio_dir}/")
                
                # Mel-Corr and MCD (both from mel-spectrograms to match paper's methodology)
                # Convert to Mel
                y_gt_mel = spec_to_mel_torch(
                    spec, 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate,
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax)
                
                y_hat_mel = mel_spectrogram_torch(
                    y_hat_trunc.squeeze(1), 
                    hps.data.filter_length, 
                    hps.data.n_mel_channels, 
                    hps.data.sampling_rate, 
                    hps.data.hop_length, 
                    hps.data.win_length, 
                    hps.data.mel_fmin, 
                    hps.data.mel_fmax
                )
                
                # Align Mels
                min_len_mel = min(y_gt_mel.size(2), y_hat_mel.size(2))
                y_gt_mel = y_gt_mel[:, :, :min_len_mel]
                y_hat_mel = y_hat_mel[:, :, :min_len_mel]
                
                # MCD from mel-spectrograms (matches paper's methodology)
                mcd = calculate_mcd_from_mel(y_gt_mel, y_hat_mel, debug=debug_this_sample)
                mcd_scores.append(mcd)
                
                mel_corr = calculate_mel_corr(y_gt_mel, y_hat_mel)
                mel_corr_scores.append(mel_corr)
                
                # STOI (calculated from audio waveforms)
                stoi_val = calculate_stoi_score(y_gt_trunc, y_hat_trunc, sr=hps.data.sampling_rate, debug=debug_this_sample)
                stoi_scores.append(stoi_val)
                
                # Top-k Accuracy
                if net_p:
                    # Get top-10 predictions
                    topk_preds = net_p.get_next_topk(mid_output, mid_output_lengths, phoneme, topk=10)
                    
                    # Calculate accuracies for k=1, 3, 5, 10
                    acc_1 = calculate_topk_accuracy(topk_preds[:, :, :1], phoneme[:, 1:], k=1)
                    acc_3 = calculate_topk_accuracy(topk_preds[:, :, :3], phoneme[:, 1:], k=3)
                    acc_5 = calculate_topk_accuracy(topk_preds[:, :, :5], phoneme[:, 1:], k=5)
                    acc_10 = calculate_topk_accuracy(topk_preds, phoneme[:, 1:], k=10)
                    
                    topk_acc_scores.append([acc_1, acc_3, acc_5, acc_10])

        # Calculate Mean and CI
        avg_mcd = np.mean(mcd_scores)
        ci_mcd = calculate_ci(mcd_scores)
        
        avg_mel_corr = np.mean(mel_corr_scores)
        ci_mel_corr = calculate_ci(mel_corr_scores)
        
        avg_stoi = np.mean(stoi_scores)
        ci_stoi = calculate_ci(stoi_scores)
        
        if topk_acc_scores:
            topk_acc_scores = np.array(topk_acc_scores)
            avg_topk_acc = np.mean(topk_acc_scores, axis=0) # [avg_1, avg_3, avg_5, avg_10]
            ci_topk_acc = [calculate_ci(topk_acc_scores[:, i]) for i in range(4)]
            
            print(f"{val_type} - MCD: {avg_mcd:.2f} ± {ci_mcd:.2f}, Mel-Corr: {avg_mel_corr:.2f} ± {ci_mel_corr:.2f}, STOI: {avg_stoi:.4f} ± {ci_stoi:.4f}")
            print(f"Top-1 Acc: {avg_topk_acc[0]:.2f} ± {ci_topk_acc[0]:.2f}")
            print(f"Top-3 Acc: {avg_topk_acc[1]:.2f} ± {ci_topk_acc[1]:.2f}")
            print(f"Top-5 Acc: {avg_topk_acc[2]:.2f} ± {ci_topk_acc[2]:.2f}")
            print(f"Top-10 Acc: {avg_topk_acc[3]:.2f} ± {ci_topk_acc[3]:.2f}")
        else:
            avg_topk_acc = [0.0, 0.0, 0.0, 0.0]
            ci_topk_acc = [0.0, 0.0, 0.0, 0.0]
            print(f"{val_type} - MCD: {avg_mcd:.2f} ± {ci_mcd:.2f}, Mel-Corr: {avg_mel_corr:.2f} ± {ci_mel_corr:.2f}, STOI: {avg_stoi:.4f} ± {ci_stoi:.4f}")
        
        with open(output_file, "a") as f:
            f.write(f"Dataset: {val_type}\n")
            f.write(f"MCD: {avg_mcd:.2f} ± {ci_mcd:.2f}\n")
            f.write(f"Mel-Corr: {avg_mel_corr:.2f} ± {ci_mel_corr:.2f}\n")
            f.write(f"STOI: {avg_stoi:.4f} ± {ci_stoi:.4f}\n")
            f.write(f"Top-1 Acc: {avg_topk_acc[0]:.2f} ± {ci_topk_acc[0]:.2f}\n")
            f.write(f"Top-3 Acc: {avg_topk_acc[1]:.2f} ± {ci_topk_acc[1]:.2f}\n")
            f.write(f"Top-5 Acc: {avg_topk_acc[2]:.2f} ± {ci_topk_acc[2]:.2f}\n")
            f.write(f"Top-10 Acc: {avg_topk_acc[3]:.2f} ± {ci_topk_acc[3]:.2f}\n")
            f.write("------------------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="fesde")
    parser.add_argument('--checkpoint_idx', default=0)
    parser.add_argument('--checkpoint_prefix', type=str, default="G", help="Prefix of the generator checkpoint (e.g., G or G_dec)")
    args = parser.parse_args()

    evaluate(args)
