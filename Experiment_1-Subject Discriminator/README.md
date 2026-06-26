# EEG-to-Speech Training and Inference

This experiment contains an EEG-to-speech pipeline that learns to map EEG signals to speech waveforms while also supervising the latent representation with phoneme prediction and subject-adversarial objectives.

The main training entry point is `train.py`. For inspecting a trained checkpoint on a single EEG sample, use `tutorial_architecture_walkthrough.ipynb`.

## Main Architecture

The model is composed of three main parts:

1. `EEGModule` in [EEGModule.py](EEGModule.py)
   - Encodes the EEG input with convolutional blocks.
   - Applies an additional temporal downsampling stage.
   - Optionally uses S4 (Structured State Space
Sequence) layers to produce the latent EEG representation `mid_output`.
   - Reconstructs the EEG signal through a decoder branch for an auxiliary reconstruction loss.
   - Includes a `subject_discriminator` on the EEG latent representation. It uses gradient reversal, so the encoder is encouraged to learn subject-invariant features.

2. `SpeechDecoder` in [models.py](models.py)
   - Takes the EEG latent representation and projects it into the speech latent space.
   - Uses a connector, flow-based latent modeling, and a HiFi-GAN style generator to synthesize waveform audio.
   - Includes a speech-side phoneme prediction head (`speech_phoneme_predictor`) that predicts phoneme sequences from the speech latent representation.
   - The current implementation also includes a speech-side subject discriminator.

3. `PhonemePredictor` in [models.py](models.py)
   - A phoneme predictor is applied to the EEG latent representation in `train.py`.
   - Another phoneme predictor is applied inside the speech module on the speech latent representation.
   - Both phoneme heads are trained with CTC loss.

During training, the model combines:

- adversarial waveform loss from `MultiPeriodDiscriminator`
- feature matching loss
- mel reconstruction loss
- KL loss
- EEG reconstruction loss
- EEG phoneme CTC loss
- speech phoneme CTC loss
- subject-adversarial loss

## Important Files

- [train.py](train.py): main training script
- [EEGModule.py](EEGModule.py): EEG encoder, reconstruction branch, subject discriminator
- [models.py](models.py): speech decoder, phoneme predictors, discriminator modules
- [configs/configs.json](configs/configs.json): default training configuration
- [config_example.json](config_example.json): example configuration template
- [inference.py](inference.py): synthesize validation examples from saved checkpoints
- [tutorial_architecture_walkthrough.ipynb](tutorial_architecture_walkthrough.ipynb): notebook to inspect a trained model on a single EEG sample

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Before training, update the paths in your config file so they match your machine:

- `data.training_files`
- `data.validation_files_both`
- `data.validation_files_audio`
- `data.validation_files_subject`
- `data.data_root_dir`
- `train.pretrained_audio`
- `train.pretrained_eeg`

If you do not want to initialize from pretrained checkpoints, set `train.pretrained_audio` and `train.pretrained_eeg` to empty strings.

The default repo config is [configs/configs.json](configs/configs.json). You can also start from [config_example.json](config_example.json).

## How to Start Training

Training is launched from `train.py`:

```bash
python train.py -c configs/configs.json -m <run_name>
```

Example:

```bash
python train.py -c configs/configs.json -m exp_subject_phoneme
```

Optional Weights and Biases logging:

```bash
python train.py -c configs/configs.json -m exp_subject_phoneme -w y
```

What this does:

- creates `logs/<run_name>/`
- copies the selected config into `logs/<run_name>/configs.json`
- writes TensorBoard logs and `train.log`
- saves checkpoints such as `G_*.pth`, `D_*.pth`, `E_*.pth`, and `P_*.pth`
- resumes automatically from the latest checkpoints in the same log directory if they already exist

## Training Notes

- `train.py` asserts that CUDA is available. CPU training is not supported.
- The script is currently wired for single-GPU CUDA execution, even though it wraps the models with DDP.
- Because the backend is `nccl`, training is expected to run in a Linux CUDA environment.

## Check a Trained Model with the Notebook

Use [tutorial_architecture_walkthrough.ipynb](tutorial_architecture_walkthrough.ipynb) to inspect the forward pass of a trained model and generate a prediction from a checkpoint.

The notebook does the following:

1. Loads a raw EEG `.npy` file.
2. Builds `EEGModule` and `SpeechDecoder` from the saved config.
3. Reads the latest `E_*.pth` and matching `G_*.pth` from the checkpoint directory.
4. Traces tensor shapes through the EEG encoder and speech decoder.
5. Runs `SpeechDecoder.infer()` to synthesize audio from the EEG sample.
6. Displays the output summary and plays the generated waveform.

To use it:

1. Open the notebook from the project root.
2. Edit the setup cell and set:
   - `EEG_FILE` to the EEG `.npy` sample you want to test
   - `CHECKPOINT_DIR` to your training run, for example `logs/<run_name>`
3. Run the notebook cells in order.

The notebook looks for `configs.json` inside the checkpoint directory first. If it is not found there, it falls back to `configs/configs.json`.

By default, the notebook automatically loads the latest available `E_*.pth` checkpoint and the matching `G_*.pth` checkpoint from the same directory. If you want to inspect a specific checkpoint instead of the latest one, edit the checkpoint-loading cell in the notebook.

## Batch Inference

If you want to synthesize validation samples outside the notebook, use [inference.py](inference.py):

```bash
python inference.py --run_name <run_name> --checkpoint_idx <step>
```

This loads `logs/<run_name>/G_<step>.pth` and `logs/<run_name>/E_<step>.pth` and writes synthesized audio under `logs/<run_name>/synthesized/<step>/`.
