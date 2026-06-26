import os
import utils
import argparse
import torch
from data_utils import (
  EEGAudioLoader,
  EEGAudioCollate
)
from torch.utils.data import DataLoader
from EEGModule import EEGModule
from models import SpeechDecoder
from scipy.io.wavfile import write

torch.manual_seed(777)

OUTPUT_DIR = './logs'

def synthesize(eeg_enc, audio_gen, eval_loader, suffix, hps, args):

    eeg_enc.eval()
    audio_gen.eval()

    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, phoneme, phoneme_lengths, sid) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            
            # x_mask_output, mid_output, eeg_decoder_output, phoneme_logits
            x, x_mask_output, mid_output, eeg_decoder_output, sid_logits = eeg_enc(x)

            mid_output_lengths = x_lengths.clone() * mid_output.size(2) / x.size(2)
            mid_output_lengths = mid_output_lengths.long()

            # Note: audio_gen is SpeechDecoder
            y_hat, _, mask, *_ = audio_gen.infer(mid_output, mid_output_lengths, max_len=1000, noise_scale=0.667)
            
            output_path = os.path.join(OUTPUT_DIR, args.run_name, 'synthesized', str(args.checkpoint_idx), suffix)
            os.makedirs(output_path, exist_ok=True)

            # y is [B, 1, T]
            write(os.path.join(output_path, f'{batch_idx}_gt.wav'), hps.data.sampling_rate, y[0,0,:y_lengths[0]].cpu().float().numpy())
            write(os.path.join(output_path, f'{batch_idx}_syn.wav'), hps.data.sampling_rate, y_hat[0,0,:y_hat.size(2)].cpu().float().numpy())

    return


def run(args):

    # Path to logs folder
    model_dir = os.path.join('./logs', args.run_name)
    config_path = os.path.join(model_dir, 'configs.json')
    
    if not os.path.exists(config_path):
        # fallback to root config if not found in logs
        config_path = "configs/configs.json"
        print(f"Warning: Config not found in {model_dir}, using {config_path}")
        
    hps = utils.get_hparams_from_file(config_path)
    
    collate_fn = EEGAudioCollate()

    eval_loader_both = DataLoader(EEGAudioLoader(hps.data.validation_files_both, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_audio = DataLoader(EEGAudioLoader(hps.data.validation_files_audio, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    eval_loader_subject = DataLoader(EEGAudioLoader(hps.data.validation_files_subject, hps.data), num_workers=1, shuffle=False,
        batch_size=1, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
    eval_loaders = [eval_loader_both, eval_loader_audio, eval_loader_subject]
    val_types = ["both", "audio", "subject"]

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

    # Load checkpoints
    checkpoint_idx = args.checkpoint_idx
    g_path = os.path.join(model_dir, f"G_{checkpoint_idx}.pth")
    e_path = os.path.join(model_dir, f"E_{checkpoint_idx}.pth")

    if not os.path.exists(g_path) or not os.path.exists(e_path):
        print(f"Error: Checkpoints not found in {model_dir}")
        print(f"Expected: {g_path} and {e_path}")
        return

    print(f"Loading checkpoints from {model_dir}...")
    utils.load_checkpoint(g_path, net_g, None)
    utils.load_checkpoint(e_path, eeg_module, None)

    for val_type, eval_loader in zip(val_types, eval_loaders):
        print(f"Synthesizing {val_type} samples...")
        synthesize(eeg_module, net_g, eval_loader, val_type, hps, args)
    
    print(f"Synthesized samples saved to: {os.path.join(OUTPUT_DIR, args.run_name, 'synthesized', str(checkpoint_idx))}")

    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help="Name of the experiment in logs folder")
    parser.add_argument('--checkpoint_idx', type=int, required=True, help="Step/Iteration index of the checkpoint")
    args = parser.parse_args()

    run(args)
