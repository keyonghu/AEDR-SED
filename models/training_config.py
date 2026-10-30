"""Model presets, local data paths and evaluation defaults."""

import json
from pathlib import Path

from config import AUDIOSET_PATHS, RESOURCES_FOLDER

MODULE_FLAGS = ("use_aedr", "aedr_residual", "use_osfnd", "use_boundary_loss")


def add_training_preset(parser):
    parser.set_defaults(
        dataset_name="audioset_strong",
        evaluation_classes=None,
        psds_dtc_threshold=0.7,
        psds_gtc_threshold=0.7,
        psds_cttc_threshold=None,
        psds_alpha_ct=0.0,
        psds_alpha_st=1.0,
        psds_max_efpr=100.0,
    )
    parser.add_argument(
        "--model_preset",
        choices=["aedr_sed", "baseline", "custom"],
        default="aedr_sed",
        help=(
            "Default: complete AEDR-SED with residual AEDR, OSFND and directional "
            "boundary loss. Select baseline or custom for component ablations."
        ),
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="Print the resolved model and paths, then exit without training or downloading.",
    )


def apply_model_preset(args):
    """Resolve unset switches; explicit positive/negative switches take priority."""
    full_model = args.model_preset == "aedr_sed"
    residual_was_unset = args.aedr_residual is None
    for flag in MODULE_FLAGS:
        if getattr(args, flag) is None:
            setattr(args, flag, full_model)
    if not args.use_aedr and residual_was_unset:
        args.aedr_residual = False
    return args


def configure_audioset_paths(args):
    """Resolve the local WAV training/development/evaluation paths."""
    fields = (
        "train_audio_dir",
        "train_annotation_file",
        "eval_audio_dir",
        "eval_annotation_file",
        "mid_to_display_name_file",
    )
    defaults = dict(AUDIOSET_PATHS)
    if not args.evaluate:
        defaults["eval_audio_dir"] = defaults["dev_audio_dir"]
        defaults["eval_annotation_file"] = defaults["dev_annotation_file"]
    for name in fields:
        if getattr(args, name) is None:
            setattr(args, name, str(defaults[name]))
    return args


def check_audioset_paths(args):
    directories = ["eval_audio_dir"]
    files = ["eval_annotation_file", "mid_to_display_name_file"]
    if not args.evaluate:
        directories.append("train_audio_dir")
        files.append("train_annotation_file")
    missing = [
        f"{name}: {getattr(args, name)}"
        for name in directories
        if not Path(getattr(args, name)).is_dir()
    ]
    missing += [
        f"{name}: {getattr(args, name)}"
        for name in files
        if not Path(getattr(args, name)).is_file()
    ]
    if missing:
        raise ValueError(
            "Prepare the dataset and configure AUDIOSET_PATHS in config.py "
            "or pass explicit path arguments. Missing paths:\n  " + "\n  ".join(missing)
        )


def print_training_config(args, dataset):
    fields = (
        "model_preset",
        "model_name",
        "pretrained",
        "beats_alignment_mode",
        *MODULE_FLAGS,
        "aedr_mode",
        "aedr_decay",
        "osfnd_warmup_epochs",
        "osfnd_min_weight",
        "boundary_mode",
        "save_model",
        "train_audio_dir",
        "train_annotation_file",
        "eval_audio_dir",
        "eval_annotation_file",
        "mid_to_display_name_file",
        "dataset_root",
        "data_backend",
    )
    values = {name: getattr(args, name) for name in fields if hasattr(args, name)}
    values.update(dataset=dataset, resources_folder=RESOURCES_FOLDER)
    print(json.dumps(values, indent=2, ensure_ascii=True))
