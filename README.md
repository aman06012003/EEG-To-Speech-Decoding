# Master's Project: EEG-to-Listened-Speech Decoding

This repository contains a master's project on decoding listened speech from EEG signals. It uses the  FESDE (Fully-End-to-end Speech Decoding from EEG) phoneme implementation as a reference baseline and adds two experiment tracks that extend the original pipeline.

## Project Overview

The project focuses on reconstructing listened speech from EEG recordings collected on the N400 dataset. The work in this repository explores two directions:

- `Experiment_1`: a modified EEG-to-speech pipeline with phoneme supervision and subject-invariant learning.
- `Experiment_2`: an extended version of the pipeline that can use the LaBraM foundation model as the EEG encoder backbone.

Each experiment folder has its own README with training and inference details.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `Source Code/icassp25-fesde-phoneme/` | Original reference implementation based on the FESDE-phoneme project. |
| `Experiment_1/` | First modified experiment built on the reference code. |
| `Experiment_2/` | Second experiment with LaBraM integration and additional utilities. |

Useful entry points:

- [`Experiment_1-/README.md`](Experiment_1-Subject_Discriminator/README.md)
- [`Experiment_2/README.md`](Experiment_2-EEG_Foundation_Model/README.md)
- [`Source Code/icassp25-fesde-phoneme/README.md`](Source%20Code/icassp25-fesde-phoneme/README.md)

## Dataset

The experiments use the N400 dataset.
Data Link: https://datadryad.org/dataset/doi:10.5061/dryad.6wwpzgmx4
- 21 subjects are included.
- Each subject listened to 440 non-prosodic audio stimuli.
- EEG signals were recorded while subjects listened to the audio.

The raw dataset is not included in this repository. Before training, you need to prepare the data locally and update the config paths to match your machine.

## Preprocessing Summary

### EEG

The EEG preprocessing pipeline used in this project is:

1. Apply a 60 Hz notch filter to remove power-line noise.
2. Apply a 0.5-50 Hz band-pass filter.
3. Remove eye-blink artifacts with Independent Component Analysis (ICA).
4. Resample the signals from 512 Hz to 256 Hz.

### Audio

Speech samples are resampled to 22,050 Hz to match the sampling setup used by the speech generation pipeline.

## Getting Started

Clone the repository:

```bash
https://github.com/aman06012003/IT_Project.git
cd IT_Project
```

Choose an experiment folder:

```bash
cd Experiment_1
```

or

```bash
cd Experiment_2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Before training:

- review `configs/configs.json`
- replace any machine-specific dataset and checkpoint paths
- confirm that your environment matches the experiment requirements

Both experiment folders are designed for CUDA-based training. Refer to the experiment-specific README before running long jobs.

## Training

From the selected experiment directory, start training with:

```bash
python train.py -c configs/configs.json -m <run_name>
```

Optional Weights & Biases logging:

```bash
python train.py -c configs/configs.json -m <run_name> -w y
```

## Instructions to start

If you are new to the repository, this is the fastest path:

1. Read the experiment README for the variant you want to run.
2. Update the config paths in that experiment folder.
3. Install the dependencies from that folder's `requirements.txt`.
4. Run `train.py` with a new run name.

For code reference -:

- `train.py` is the main training entry point in each experiment folder.
- `inference.py` is used to synthesize outputs from saved checkpoints.
- `tutorial_architecture_walkthrough.ipynb` helps inspect model behavior on a single EEG sample.

# Acknowledgement
For this project we thank the authors of [1,2] to create an intial baseline for decoding speech from raw EEG. We also want to thank the authors of [3] for providing the dataset of participants.

## References

[1]. Lee, Jihwan, et al. "Toward fully-end-to-end listened speech decoding from EEG signals." arXiv preprint arXiv:2406.08644, 2024..<br>
[2]. Lee, Jihwan, et al. "Enhancing listened speech decoding from EEG via parallel phoneme sequence prediction." ICASSP 2025 IEEE International Conference on Acoustics, Speech and Signal Processing, 2025..<br>
[3]. Toffolo, Kathryn K., Edward G. Freedman, and John J. Foxe. "Evoking the N400 event-related potential (ERP) component using a publicly available novel set of sentences with semantically incongruent or congruent endings." Neuroscience 501, 2022, pp. 143-158..<br>


