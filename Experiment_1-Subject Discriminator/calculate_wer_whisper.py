"""
Calculate Word Error Rate (WER) using OpenAI Whisper on synthesized speech from EEG.
This evaluates the intelligibility of the EEG-to-Speech system.
"""

import os
import argparse
import torch
import numpy as np
import librosa
import whisper
import jiwer
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from data_utils import EEGAudioLoader, EEGAudioCollate
from EEGModule import EEGModule
from models import SpeechDecoder

def normalize_text(text):
    """Basic text normalization for WER calculation"""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text


def evaluate_wer(args):
    # Load config and settings
    logs_dir = Path(args.logs_dir).resolve()
    run_name = args.run_name

    print(f"--- DEBUG INFO ---")
    print(f"Target Run: {run_name}")
    print(f"Base Logs Dir: {logs_dir}")

    # 1. Primary Check: logs_dir / run_name / configs.json
    config_path = logs_dir / run_name / 'configs.json'
    print(f"Attempt 1 (Primary): {config_path}")

    if not config_path.exists():
        # 2. Secondary Check: Local logs folder
        local_logs = Path('./logs').resolve()
        config_path = local_logs / run_name / 'configs.json'
        print(f"Attempt 2 (Local Fallback): {config_path}")

    if not config_path.exists():
        # List what IS there to help the user
        print(f"FAILED to find configs.json")
        if logs_dir.exists():
            print(f"List of directories in {logs_dir}:")
            for d in logs_dir.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        else:
            print(f"Error: {logs_dir} does not exist or is not a directory.")

        raise FileNotFoundError(f"Could not find configs.json for run '{run_name}' in either {args.logs_dir} or ./logs/")

    print(f"SUCCESS: Found config at {config_path}")
    print(f"--- END DEBUG ---\n")

    hps = utils.get_hparams_from_file(str(config_path))
    sampling_rate = hps.data.sampling_rate

    # Load Whisper model (16kHz expected)
    print(f"Loading Whisper {args.whisper_model} model...")
    whisper_model = whisper.load_model(args.whisper_model)

    # Setup Data Loader
    collate_fn = EEGAudioCollate()
    val_file = hps.data.validation_files_subject if args.split == 'subject' else \
               hps.data.validation_files_both if args.split == 'both' else \
               hps.data.validation_files_audio

    print(f"Loading dataset from: {val_file}")
    dataset = EEGAudioLoader(val_file, hps.data)
    loader = DataLoader(dataset, num_workers=1, shuffle=False,
                         batch_size=1, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    # Initialize Models
    eeg_module = EEGModule(
        n_layers_cnn=hps.model.eeg_module.n_layers_cnn,
        use_s4=hps.model.eeg_module.use_s4,
        n_layers_s4=hps.model.eeg_module.n_layers_s4,
        embedding_size=hps.model.inter_channels,
        is_mask=False,
        in_channels=hps.model.eeg_module.in_channels,
        num_subjects=hps.model.num_subjects if hasattr(hps.model, 'num_subjects') else 25
    ).cuda()

    net_g = SpeechDecoder(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()

    # Load checkpoints - use same parent directory as configs.json
    checkpoint_idx = args.checkpoint_idx
    checkpoint_base = config_path.parent

    g_path = checkpoint_base / f"G_{checkpoint_idx}.pth"
    e_path = checkpoint_base / f"E_{checkpoint_idx}.pth"

    if not e_path.exists():
        # Try E_dec_ prefix fallback
        alt_e_path = checkpoint_base / f"E_dec_{checkpoint_idx}.pth"
        if alt_e_path.exists():
            e_path = alt_e_path

    print(f"Loading Generator: {g_path}")
    print(f"Loading EEG Module: {e_path}")

    utils.load_checkpoint(str(g_path), net_g, None)
    utils.load_checkpoint(str(e_path), eeg_module, None)

    eeg_module.eval()
    net_g.eval()

    print(f"Evaluating {args.split} split for {run_name} at step {checkpoint_idx}...")

    ground_truths = []
    transcriptions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            # Handle different batch sizes or tuple formats
            if len(batch) == 9:
                x, x_lengths, spec, spec_lengths, y, y_lengths, phoneme, phoneme_lengths, sid = batch
            else:
                x, x_lengths, spec, spec_lengths, y, y_lengths, phoneme, phoneme_lengths = batch

            x, x_lengths = x.cuda(0), x_lengths.cuda(0)

            # Get ground truth text from the dataset entry
            raw_entry = dataset.audiopaths_and_eeg[batch_idx]
            _, target_text, _ = raw_entry.split('||')

            # Forward Pipeline
            _, _, mid_output, _, _ = eeg_module(x)
            mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
            mid_output_lengths = mid_output_lengths.long()

            # Synthesize Audio (Standard Inference)
            # noise_scale=0 for deterministic evaluation
            y_hat, *_ = net_g.infer(mid_output, mid_output_lengths, max_len=None, noise_scale=0.0)

            # Convert audio to numpy and resample to 16kHz for Whisper
            audio = y_hat[0, 0].cpu().numpy()
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

            # Transcribe with Whisper
            result = whisper_model.transcribe(audio, fp16=torch.cuda.is_available())
            pred_text = result["text"]

            # Store normalized text
            gt_norm = normalize_text(target_text)
            pred_norm = normalize_text(pred_text)

            ground_truths.append(gt_norm)
            transcriptions.append(pred_norm)

            if batch_idx < 5: # Debug print
                print(f"\nSample {batch_idx}:")
                print(f"  Target:  {gt_norm}")
                print(f"  Whisper: {pred_norm}")

    # Calculate final WER
    wer = jiwer.wer(ground_truths, transcriptions)
    mer = jiwer.mer(ground_truths, transcriptions)
    wil = jiwer.wil(ground_truths, transcriptions)

    print("\n" + "="*30)
    print(f"Results for {run_name} ({args.split}):")
    print(f"WER: {wer*100:.2f}%")
    print(f"MER: {mer*100:.2f}%")
    print(f"WIL: {wil*100:.2f}%")
    print("="*30)

    # Save results
    output_path = f"wer_whisper_{run_name}_{checkpoint_idx}_{args.split}.txt"
    with open(output_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_idx}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Whisper Model: {args.whisper_model}\n")
        f.write(f"WER: {wer*100:.2f}%\n")
        f.write(f"MER: {mer*100:.2f}%\n")
        f.write(f"WIL: {wil*100:.2f}%\n")
        f.write("\nDetailed Transcriptions:\n")
        for gt, pr in zip(ground_truths, transcriptions):
            f.write(f"Ref: {gt}\n")
            f.write(f"Hyp: {pr}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--checkpoint_idx', type=int, required=True)
    parser.add_argument('--logs_dir', type=str, default='/media/hdd3/amkapoor/logs', help='Path to directory containing logs')
    parser.add_argument('--split', type=str, default='subject', choices=['both', 'audio', 'subject'])
    parser.add_argument('--whisper_model', type=str, default='base')
    args = parser.parse_args()

    evaluate_wer(args)
