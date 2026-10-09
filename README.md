# Motor Intention Decoding from EEG

Research scripts for **left-hand, right-hand and rest motor-imagery EEG decoding** developed during a UROP project.

## Repository structure

- `offline/preprocessing/` — preprocessing and data preparation.
- `offline/classification/` — baseline classifiers, FBCSP and hierarchical methods.
- `offline/visualization/` — PSD, time–frequency, topographic and null-space analyses.
- `online/model_training/` — experiments for exporting an online decoder.
- `online/matlab/` — MATLAB model training, offline evaluation and stream-handling helpers.
- `online/README.md` — status and limitations of the online work.

## Research workflow

EEG motor-imagery decoding experiments, including filtering, epoch/window preparation, feature extraction with CSP/FBCSP, and classification with algorithms such as LDA and SVM. These are research scripts rather than a packaged or independently validated pipeline.

**Important:** Some scripts were used for offline experiments and model-development work, not successful real-time decoding. The separate online-test ZIP was provided but could not be extracted during this repository import. It is **not** represented as a complete or validated real-time application.

## Reproducing experiments

Review the path, channel-selection, sampling-rate and recording-format assumptions in each script before running. Dataset files and generated trained models are not included. Python dependencies vary by script and include NumPy, SciPy, MNE-Python, scikit-learn and plotting libraries. MATLAB scripts may require additional signal-processing/toolbox functions and the project's original acquisition environment.

## Data and privacy

Do not commit identifiable EEG recordings, participant metadata, generated models containing sensitive information or local environment configuration. See `.gitignore`.

## Status

Archived research code. Scripts are preserved for traceability, and no claim is made that every script runs as-is without the original data/environment.
