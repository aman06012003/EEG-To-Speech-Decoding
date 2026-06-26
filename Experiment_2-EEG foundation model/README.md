# EEG-to-Audio Training and Inference

In this experiment we integrated LaBraM foundational model in EEG encoder to generate EEG embeddings. It learns a latent representation from EEG, aligns that latent with a speech decoder, and generates waveform audio from the predicted speech representation.

## Architecture

The training pipeline is built from three main components:

1. `EEGModule`
   - Defined in `EEGModule.py`.
   - Encodes EEG into a latent sequence `mid_output`.
   - Supports two EEG backbones:
     - A CNN + optional S4 (Structured State Space Sequence) stack.
     - A LaBraM backbone (`use_labram: true` in the config).
   - Includes an EEG reconstruction branch used as an auxiliary loss.
   - Includes a subject discriminator for subject-invariant latent learning.

2. `SpeechDecoder`
   - Defined in `models.py`.
   - Projects the EEG latent into the speech latent space through `enc_proj`.
   - Uses a VITS-style latent flow and waveform generator to synthesize speech.
   - Can also include a phoneme predictor head.

3. `MultiPeriodDiscriminator`
   - Defined in `models.py`.
   - Adversarial discriminator used during waveform training.

During training in `train.py`, the model optimizes a combination of:
- adversarial loss
- feature matching loss
- mel reconstruction loss
- KL loss
- EEG reconstruction loss
- CTC phoneme loss
- subject adversarial loss

## Data Layout

The default config expects:

- EEG `.npy` files under `N400_epoched/...`
- stimulus `.wav` files under `stimuli/...`
- train and validation filelists under `filelists/n400/...`

Each filelist row is expected to contain:

```text
relative/eeg_path||text||phonemes
```

The loader in `data_utils.py` derives:
- the EEG path from `N400_epoched/<datapath>.wav.npy`
- the audio path from `stimuli/<stimulus_name>.wav`

## Configuration

The main experiment config is [configs/configs.json](/e:/Master_thesis/Experiment_2/configs/configs.json).

Before training, review these fields carefully:

- `data.data_root_dir`
- `data.training_files`
- `data.validation_files_both`
- `data.validation_files_audio`
- `data.validation_files_subject`
- `model.eeg_module.use_labram`
- `model.eeg_module.labram_checkpoint`
- `train.pretrained_audio`
- `train.pretrained_eeg`

The checked-in config currently contains machine-specific absolute paths, so you will need to replace them with paths valid on your system before running training.

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- Training requires CUDA. `train.py` explicitly asserts that CPU training is not allowed.
- If you use the LaBraM backbone with DoRA/LoRA, make sure `peft` is installed.

## Start Training

Training is launched from [train.py](/e:/Master_thesis/Experiment_2/train.py). The script uses:

- `-c` or `--config` for the config JSON
- `-m` or `--model` for the run name
- `-w` or `--wandb` to enable Weights & Biases logging

Example:

```bash
python train.py -c configs/configs.json -m my_run
```

With Weights & Biases enabled:

```bash
python train.py -c configs/configs.json -m my_run -w y
```

This creates:

```text
logs/my_run/
```

Inside that directory the training script stores:
- `configs.json`
- TensorBoard logs
- checkpoints such as `G_<step>.pth`, `D_<step>.pth`, `E_<step>.pth`, and `P_<step>.pth`

Checkpoint meanings:
- `G_*`: speech generator / decoder
- `D_*`: discriminator
- `E_*`: EEG encoder module
- `P_*`: phoneme predictor

## Run Prediction From a Trained Checkpoint

To synthesize audio from saved checkpoints, use [inference.py](/e:/Master_thesis/Experiment_2/inference.py):

```bash
python inference.py --run_name my_run --checkpoint_idx 224000
```

This expects the following files to exist:

```text
logs/my_run/G_224000.pth
logs/my_run/E_224000.pth
```

Generated audio is written to:

```text
logs/my_run/synthesized/224000/
```

with separate outputs for:
- `both`
- `audio`
- `subject`

Each folder contains paired files such as:
- `0_gt.wav`
- `0_syn.wav`

## Notebook Walkthrough

Use [tutorial_architecture_walkthrough.ipynb](/e:/Master_thesis/Experiment_2/tutorial_architecture_walkthrough.ipynb) to inspect how a trained model produces predictions from EEG using saved checkpoints.

The notebook is useful for:
- loading a single EEG sample
- loading the experiment config
- restoring the latest or a selected `E_*.pth` and `G_*.pth` checkpoint
- tracing tensor shapes through the EEG encoder and speech decoder
- generating and listening to synthesized audio

Before running the notebook, update the path variables inside it, especially:
- `EEG_FILE`
- `CHECKPOINT_DIR`
- optional fallback `CONFIG_PATH`

The notebook looks for checkpoints in the chosen run directory and then loads:
- `E_<step>.pth`
- `G_<step>.pth`

This is the recommended place to verify the prediction of a trained model from a checkpoint, because it shows both the architecture flow and the final synthesized output step by step.

## Relevant Files

- [train.py](/e:/Master_thesis/Experiment_2/train.py): main training entrypoint
- [inference.py](/e:/Master_thesis/Experiment_2/inference.py): checkpoint-based synthesis script
- [EEGModule.py](/e:/Master_thesis/Experiment_2/EEGModule.py): EEG encoder and reconstruction branch
- [models.py](/e:/Master_thesis/Experiment_2/models.py): speech decoder, discriminator, and auxiliary modules
- [data_utils.py](/e:/Master_thesis/Experiment_2/data_utils.py): dataset and collate logic
- [configs/configs.json](/e:/Master_thesis/Experiment_2/configs/configs.json): experiment configuration
- [tutorial_architecture_walkthrough.ipynb](/e:/Master_thesis/Experiment_2/tutorial_architecture_walkthrough.ipynb): checkpoint walkthrough notebook
