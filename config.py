from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# Configure local datasets and the pretrained-weight cache here once.
DATA_ROOT = PROJECT_ROOT / "data"
DATASET = "audioset"
RESOURCES_FOLDER = str(PROJECT_ROOT / "checkpoints" / "pretrained")
DESED_ROOT = DATA_ROOT / "desed"
AUDIOSET_PATHS = {
    "train_audio_dir": DATA_ROOT / "audioset/audio/train",
    "train_annotation_file": DATA_ROOT / "audioset/metadata/train_strong.tsv",
    "dev_audio_dir": DATA_ROOT / "audioset/audio/train",
    "dev_annotation_file": DATA_ROOT / "audioset/metadata/dev_strong.tsv",
    "eval_audio_dir": DATA_ROOT / "audioset/audio/eval",
    "eval_annotation_file": DATA_ROOT / "audioset/metadata/eval_strong.tsv",
    "mid_to_display_name_file": DATA_ROOT / "audioset/metadata/mid_to_display_name.tsv",
}
GITHUB_RELEASE_URL_TRANSFORMERS = (
    "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"
)

# Default training uses BEATs_weak. SSL/strong variants are optional encoder
# initializations, not additional model architectures.
CHECKPOINT_URLS = {
    name: GITHUB_RELEASE_URL_TRANSFORMERS + name + ".pt"
    for name in ("BEATs_weak", "BEATs_ssl", "BEATs_strong_1")
}
