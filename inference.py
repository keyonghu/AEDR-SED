"""Run a trained BEATs-based AEDR-SED checkpoint on one audio recording."""

import argparse
from pathlib import Path

import torch

from models.class_names import DESED_SYNTHETIC_CLASSES, as_strong_train_classes
from models.prediction_wrapper import PredictionsWrapper
from models.transformers.beats.BEATs_wrapper import BEATsWrapper
from utils import ManyHotEncoder, batched_decode_preds


def load_model(args):
    """Load a trained state dictionary or Lightning checkpoint."""
    backbone = BEATsWrapper(alignment_mode=args.beats_alignment_mode)
    labels = (
        DESED_SYNTHETIC_CLASSES if args.dataset == "desed" else as_strong_train_classes
    )
    model = PredictionsWrapper(
        backbone,
        checkpoint=None,
        embed_dim=768,
        n_classes_strong=len(labels),
        aedr_enabled=args.use_aedr,
        aedr_mode=args.aedr_mode,
        aedr_decay=args.aedr_decay,
        aedr_kernel_size=args.aedr_kernel_size,
        aedr_residual=args.aedr_residual,
        aedr_residual_scale_init=args.aedr_residual_scale_init,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in payload:
        # Exclude training-only loss buffers and strip the Lightning prefix.
        state = {
            key[len("model.") :]: value
            for key, value in payload["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        state = payload
    model.load_state_dict(state, strict=True)
    return model, labels


def sound_event_detection(args):
    import librosa

    device_name = "cuda" if args.cuda else args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)
    model, labels = load_model(args)
    model.to(device).eval()

    waveform, _ = librosa.load(args.audio_file, sr=16_000, mono=True)
    if len(waveform) == 0:
        raise ValueError("The audio file is empty.")
    duration = len(waveform) / 16_000
    waveform = torch.from_numpy(waveform).unsqueeze(0).to(device)
    predictions = []
    # Match the ten-second training window; AEDR state resets per window.
    with torch.inference_mode():
        for start in range(0, waveform.shape[1], 160_000):
            chunk = waveform[:, start : start + 160_000]
            chunk = torch.nn.functional.pad(chunk, (0, 160_000 - chunk.shape[1]))
            logits, _ = model(model.mel_forward(chunk))
            predictions.append(logits.cpu())
    probabilities = torch.sigmoid(torch.cat(predictions, dim=2))
    encoder = ManyHotEncoder(labels, audio_len=duration)
    _, _, decoded = batched_decode_preds(
        probabilities,
        [str(args.audio_file)],
        encoder,
        thresholds=args.detection_thresholds,
        median_filter=args.median_window,
    )
    for threshold, events in decoded.items():
        events = events[events["onset"] < duration].copy()
        events["offset"] = events["offset"].clip(upper=duration)
        print(f"\nDetected events at threshold {threshold}:")
        print(events.sort_values("onset").to_string(index=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["audioset", "desed"], default="audioset")
    parser.add_argument(
        "--beats_alignment_mode",
        choices=["legacy_flatten", "temporal_mean"],
        default="legacy_flatten",
    )
    parser.add_argument(
        "--use_aedr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--aedr_mode", choices=["legacy", "reset_ema"], default="reset_ema"
    )
    parser.add_argument("--aedr_decay", type=float, default=0.95)
    parser.add_argument("--aedr_kernel_size", type=int, default=15)
    parser.add_argument(
        "--aedr_residual", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--aedr_residual_scale_init", type=float, default=0.1)
    parser.add_argument("--detection_thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--median_window", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda", action="store_true", help="Alias for --device=cuda.")
    args = parser.parse_args(argv)
    if args.aedr_residual is None:
        args.aedr_residual = args.use_aedr
    if not args.audio_file.is_file():
        parser.error(f"Audio file does not exist: {args.audio_file}")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.aedr_residual and not args.use_aedr:
        parser.error("--aedr_residual requires --use_aedr.")
    if args.median_window <= 0 or args.median_window % 2 == 0:
        parser.error("--median_window must be a positive odd integer.")
    if any(not 0 < threshold < 1 for threshold in args.detection_thresholds):
        parser.error("Detection thresholds must be strictly between zero and one.")
    return args


if __name__ == "__main__":
    sound_event_detection(parse_args())
