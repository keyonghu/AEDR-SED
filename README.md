# AEDR-SED

Research implementation of **AEDR-SED: Acoustic Edge-Driven Reset with
Reliability-Aware Supervision for Sound Event Detection**.

The model combines a BEATs encoder, residual AEDR, OSFND and a directional
boundary loss.

## Structure

```text
AEDR-SED/
├── checkpoints/       # Downloaded pretrained and newly trained weights
├── data/              # Locally prepared AudioSet Strong or DESED data
├── logs/              # Runtime logs
├── models/            # BEATs, AEDR, losses and shared training engine
├── config.py          # Dataset selection, paths and public checkpoint URLs
├── data_utils.py      # Local WAV datasets, preprocessing and sampling
├── evaluation.py      # Dataset evaluation and event metrics
├── inference.py       # Inference on an individual audio recording
├── train.py           # Training entry point
├── utils.py           # Label encoding and prediction decoding
├── requirements.txt
├── README.md
├── ACKNOWLEDGEMENTS.md
└── LICENSE
```

## Setup

Use Python 3.11 and install compatible PyTorch, torchvision and torchaudio
builds for your platform, then run:

```bash
python -m pip install --upgrade "numpy<2" "setuptools<81" wheel "Cython>=3,<4"
python -m pip install --no-build-isolation --no-deps "sed_scores_eval==0.0.3"
python -m pip install -r requirements.txt
python -m pip check
```

The pinned `sed_scores_eval` source release imports NumPy and Cython during
its build without declaring them as isolated-build dependencies. Install these
tools first and build that package without isolation; the final requirements
command then installs its runtime dependencies. This preserves the evaluation
package version used by this project. Windows source builds also require
Microsoft C++ Build Tools with an MSVC compiler and Windows SDK. Linux builds
require an available C/C++ compiler toolchain. On Windows, replace
`python` with `.\.venv\Scripts\python.exe` when using the local virtual environment.

Prepare the dataset separately and configure `DATASET`, `AUDIOSET_PATHS`
or `DESED_ROOT` in `config.py`. Missing BEATs initialization weights are
downloaded automatically to `checkpoints/pretrained/` when the configured
download address is accessible. For offline use, place `BEATs_weak.pt` in
that directory beforehand, or configure `RESOURCES_FOLDER`.
Public checkpoint URLs are listed in `config.py`. These initialize BEATs;
trained AEDR-SED checkpoints are produced by training, not downloaded.

AudioSet WAV basenames must match segment IDs. Annotation columns are
`segment_id`, `start_time_seconds`, `end_time_seconds`, and `label` (MID).
The MID/name mapping is a headerless two-column TSV. Use separate training,
development and evaluation annotations. DESED uses its standard synthetic
soundscape layout and `filename`, `onset`, `offset`, `event_label` columns.
The AudioSet loader also accepts the official headerless four-column TSVs.

Online preprocessing converts audio to mono, resamples it to 16 kHz, and
pads/truncates it to 10 seconds. The BEATs front end computes 128-bin Kaldi
filter-bank features with a 25-ms window and 10-ms hop. Strong labels are
encoded on a 25-Hz grid. No offline spectrogram cache is required.

## Train and evaluate

After configuring the data, train on the selected dataset:

```bash
python train.py
```

Or explicitly select DESED:

```bash
python train.py --dataset desed
```

Training saves checkpoints under `checkpoints/<experiment_name>/` and uses
offline logging. Evaluate the selected checkpoint with:

```bash
python evaluation.py --checkpoint /path/to/selected.ckpt
```

Use the same dataset selection and model settings as training. For audio-file
inference with a residual AEDR checkpoint:

```bash
python inference.py --audio_file /path/to/audio.wav \
  --checkpoint /path/to/selected.ckpt
```

Residual AEDR is enabled by default for inference. Supply a trained checkpoint.
For DESED, add `--dataset desed` to evaluation and inference. Use the same
AEDR settings as training when they differ from the defaults.

The repository contains training, evaluation and single-recording inference.
Datasets, weights, logs and metric results are obtained or produced separately.
Source attribution and license notices are in `ACKNOWLEDGEMENTS.md` and `LICENSE`.
