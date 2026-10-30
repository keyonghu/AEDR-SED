"""Local WAV datasets, online preprocessing and training sampling for AEDR-SED."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DistributedSampler, WeightedRandomSampler


def read_audioset_annotations(path):
    """Read normalized tables or official headerless four-column TSVs."""
    path = Path(path)
    columns = ["segment_id", "start_time_seconds", "end_time_seconds", "label"]
    table = pd.read_csv(
        path,
        sep="\t" if path.suffix.lower() == ".tsv" else ",",
        header=None,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if table.empty or table.shape[1] != len(columns):
        raise ValueError(f"Expected four AudioSet annotation columns in {path}.")
    first_row = [value.strip() for value in table.iloc[0]]
    if set(first_row) == set(columns):
        table.columns = first_row
        table = table.iloc[1:].copy()[columns]
    else:
        table.columns = columns
    # Keep the timestamp precision used for rasterizing strong labels.
    for name in ("start_time_seconds", "end_time_seconds"):
        table[name] = pd.to_numeric(table[name], errors="raise").astype("float32")
    return table.reset_index(drop=True)


def _merge_overlapping_events(events):
    """Merge overlapping intervals of the same class."""
    merged = []
    for event_label, class_events in events.groupby("event_label"):
        class_events = class_events.sort_values("onset")
        current = None
        for row in class_events.itertuples(index=False):
            onset = float(row.onset)
            offset = float(row.offset)
            if current is None or current["offset"] < onset:
                current = {
                    "onset": onset,
                    "offset": offset,
                    "event_label": event_label,
                }
                merged.append(current)
            else:
                current["offset"] = max(current["offset"], offset)
    return pd.DataFrame(merged, columns=["onset", "offset", "event_label"])


class LocalWavStrongDataset(Dataset):
    """AudioSet Strong WAV dataset backed by the public strong-label TSV files.

    The label encoder determines the output class order. For a public
    ``BEATs_strong_1`` checkpoint, pass an encoder initialized with
    ``as_strong_train_classes`` (447 classes). Evaluation-only classes that
    are not present in that list are ignored.
    """

    def __init__(
        self,
        audio_dir,
        annotations_tsv,
        mid_to_display_name_tsv,
        label_encoder,
        sample_rate=16_000,
        audio_length=10.0,
    ):
        self.audio_dir = Path(audio_dir)
        self.annotations_tsv = Path(annotations_tsv)
        self.mid_to_display_name_tsv = Path(mid_to_display_name_tsv)
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate
        self.audio_length = audio_length
        self.target_num_samples = int(sample_rate * audio_length)

        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory does not exist: {self.audio_dir}")
        if not self.annotations_tsv.is_file():
            raise FileNotFoundError(
                f"Annotation TSV does not exist: {self.annotations_tsv}"
            )
        if not self.mid_to_display_name_tsv.is_file():
            raise FileNotFoundError(
                f"MID-to-name TSV does not exist: {self.mid_to_display_name_tsv}"
            )

        mapping_separator = (
            "\t" if self.mid_to_display_name_tsv.suffix.lower() == ".tsv" else ","
        )
        mid_df = pd.read_csv(
            self.mid_to_display_name_tsv,
            sep=mapping_separator,
            header=None,
            names=["mid", "display_name"],
        )
        mid_to_name = dict(zip(mid_df["mid"], mid_df["display_name"]))

        print(f"Reading strong annotations: {self.annotations_tsv}", flush=True)
        annotations = read_audioset_annotations(self.annotations_tsv)
        annotations = annotations.rename(
            columns={
                "start_time_seconds": "onset",
                "end_time_seconds": "offset",
            }
        )
        annotations["filename"] = annotations["segment_id"].map(
            lambda value: (
                str(value) if str(value).lower().endswith(".wav") else f"{value}.wav"
            )
        )
        annotations["event_label"] = annotations["label"].map(mid_to_name)

        known_labels = set(label_encoder.labels)
        unsupported_mask = annotations["event_label"].isna() | ~annotations[
            "event_label"
        ].isin(known_labels)
        unsupported_mids = set(annotations.loc[unsupported_mask, "label"].unique())
        annotations = annotations[
            annotations["event_label"].isin(known_labels)
            & (annotations["offset"] > annotations["onset"])
        ].drop_duplicates()

        print(f"Scanning local WAV files: {self.audio_dir}", flush=True)
        available_files = {
            path.name: path for path in self.audio_dir.glob("*.wav") if path.is_file()
        }
        annotated_filenames = set(annotations["filename"].unique())
        self.missing_audio_count = len(annotated_filenames.difference(available_files))

        annotations = annotations[annotations["filename"].isin(available_files)]
        annotations = annotations.sort_values(
            ["filename", "event_label", "onset"]
        ).reset_index(drop=True)

        events_per_file = annotations.groupby("filename", sort=False).size()
        self.audio_paths = available_files
        self.filenames = events_per_file.index.tolist()
        self.event_offsets = (
            events_per_file.cumsum().shift(fill_value=0).astype(int).tolist()
        )
        self.event_ends = events_per_file.cumsum().astype(int).tolist()
        self.events = annotations[["onset", "offset", "event_label"]]
        self.unsupported_mids = sorted(unsupported_mids)

        if not self.filenames:
            raise RuntimeError(
                "No labeled WAV files matched the annotation TSV. Check the audio "
                "directory and segment_id naming."
            )

        print(
            "Loaded local AudioSet Strong data: "
            f"{len(self.filenames)} labeled WAV files, "
            f"{self.missing_audio_count} annotated files missing on disk, "
            f"{len(self.unsupported_mids)} unsupported MIDs ignored."
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        audio_path = self.audio_paths[filename]
        audio, source_sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio).mean(dim=1)

        if source_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, source_sample_rate, self.sample_rate
            )

        unpadded_num_samples = min(len(audio), self.target_num_samples)
        duration = unpadded_num_samples / self.sample_rate
        if len(audio) < self.target_num_samples:
            audio = torch.nn.functional.pad(
                audio, (0, self.target_num_samples - len(audio))
            )
        else:
            audio = audio[: self.target_num_samples]

        events = self.events.iloc[self.event_offsets[index] : self.event_ends[index]]
        events = _merge_overlapping_events(events)
        strong = torch.from_numpy(self.label_encoder.encode_strong_df(events).T).float()
        gt_string = "++".join(
            f"{row.onset};;{row.offset};;{row.event_label}"
            for row in events.itertuples(index=False)
        )

        return {
            "audio": audio,
            "strong": strong,
            "event_count": strong.sum(dim=1),
            "filename": filename,
            "gt_string": gt_string,
            "duration": duration,
        }


class LocalDesedStrongDataset(Dataset):
    """DESED synthetic subset backed by its official soundscapes TSV."""

    def __init__(
        self,
        audio_dir,
        annotations_tsv,
        label_encoder,
        sample_rate=16_000,
        audio_length=10.0,
    ):
        self.audio_dir = Path(audio_dir)
        self.annotations_tsv = Path(annotations_tsv)
        self.label_encoder = label_encoder
        self.sample_rate = int(sample_rate)
        self.audio_length = float(audio_length)
        self.target_num_samples = int(self.sample_rate * self.audio_length)

        if not self.audio_dir.is_dir():
            raise FileNotFoundError(
                f"DESED audio directory does not exist: {self.audio_dir}"
            )
        if not self.annotations_tsv.is_file():
            raise FileNotFoundError(
                f"DESED annotation TSV does not exist: {self.annotations_tsv}"
            )

        print(f"Reading DESED annotations: {self.annotations_tsv}", flush=True)
        annotations = pd.read_csv(
            self.annotations_tsv,
            sep="\t",
            dtype={"filename": str, "event_label": str},
        )
        required = {"filename", "onset", "offset", "event_label"}
        missing = required.difference(annotations.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {self.annotations_tsv}: {sorted(missing)}"
            )

        annotations = annotations[list(required)].copy()
        annotations["filename"] = annotations["filename"].map(
            lambda value: (
                str(value) if str(value).lower().endswith(".wav") else f"{value}.wav"
            )
        )
        annotations["onset"] = pd.to_numeric(annotations["onset"], errors="coerce")
        annotations["offset"] = pd.to_numeric(annotations["offset"], errors="coerce")

        known_labels = set(label_encoder.labels)
        unsupported = sorted(
            set(annotations["event_label"].dropna()).difference(known_labels)
        )
        if unsupported:
            raise ValueError(
                "DESED annotations contain labels outside the configured class "
                f"order: {unsupported}"
            )

        annotations = annotations[
            annotations["event_label"].isin(known_labels)
            & annotations["onset"].notna()
            & annotations["offset"].notna()
            & (annotations["offset"] > annotations["onset"])
        ].drop_duplicates()

        print(f"Scanning DESED WAV files: {self.audio_dir}", flush=True)
        self.audio_paths = {
            path.name: path for path in self.audio_dir.glob("*.wav") if path.is_file()
        }
        if not self.audio_paths:
            raise RuntimeError(f"No WAV files found in {self.audio_dir}")

        annotated_filenames = set(annotations["filename"])
        missing_audio = sorted(annotated_filenames.difference(self.audio_paths))
        if missing_audio:
            raise RuntimeError(
                f"{len(missing_audio)} annotated DESED WAV files are missing. "
                f"Examples: {missing_audio[:5]}"
            )

        # Include every waveform, including a possible background-only file.
        self.filenames = sorted(self.audio_paths)
        annotations = annotations.sort_values(
            ["filename", "event_label", "onset", "offset"]
        ).reset_index(drop=True)
        self.events_by_filename = {
            filename: group[["onset", "offset", "event_label"]].reset_index(drop=True)
            for filename, group in annotations.groupby("filename", sort=False)
        }
        self.empty_events = pd.DataFrame(columns=["onset", "offset", "event_label"])
        self.balanced_sample_weights = self._balanced_weights(annotations)

        annotated_count = sum(
            filename in self.events_by_filename for filename in self.filenames
        )
        print(
            "Loaded DESED synthetic soundscapes: "
            f"{len(self.filenames)} WAV files, {annotated_count} files with "
            f"events, {len(label_encoder.labels)} classes.",
            flush=True,
        )

    def _balanced_weights(self, annotations, sample_weight_offset=30.0):
        frame_rate = (
            self.label_encoder.fs
            / self.label_encoder.frame_hop
            / self.label_encoder.net_pooling
        )
        clipped_onsets = annotations["onset"].clip(0.0, self.audio_length)
        clipped_offsets = annotations["offset"].clip(0.0, self.audio_length)
        duration_frames = (clipped_offsets - clipped_onsets).clip(
            lower=0.0
        ) * frame_rate
        weighted = annotations[["filename", "event_label"]].copy()
        weighted["duration_frames"] = duration_frames
        temporal_counts = weighted.groupby("event_label")["duration_frames"].sum()
        class_weights = 1000.0 / (temporal_counts + sample_weight_offset)
        weighted["weight"] = weighted["duration_frames"] * weighted["event_label"].map(
            class_weights
        )
        sample_weights = (
            weighted.groupby("filename")["weight"].sum().reindex(self.filenames)
        )
        positive = sample_weights[sample_weights > 0]
        floor = float(positive.min()) * 0.01 if not positive.empty else 1.0
        return torch.tensor(
            sample_weights.fillna(floor).clip(lower=floor).to_numpy(),
            dtype=torch.float32,
        )

    def get_balanced_sample_weights(self):
        return self.balanced_sample_weights

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        audio, source_sample_rate = sf.read(
            self.audio_paths[filename], dtype="float32", always_2d=True
        )
        audio = torch.from_numpy(audio).mean(dim=1)
        if source_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio, source_sample_rate, self.sample_rate
            )

        unpadded_samples = min(len(audio), self.target_num_samples)
        duration = unpadded_samples / self.sample_rate
        if len(audio) < self.target_num_samples:
            audio = torch.nn.functional.pad(
                audio, (0, self.target_num_samples - len(audio))
            )
        else:
            audio = audio[: self.target_num_samples]

        events = self.events_by_filename.get(filename, self.empty_events)
        events = _merge_overlapping_events(events)
        strong = torch.from_numpy(self.label_encoder.encode_strong_df(events).T).float()
        gt_string = "++".join(
            f"{row.onset};;{row.offset};;{row.event_label}"
            for row in events.itertuples(index=False)
        )
        return {
            "audio": audio,
            "strong": strong,
            "event_count": strong.sum(dim=1),
            "filename": filename,
            "gt_string": gt_string,
            "duration": duration,
        }


def get_uniform_sample_weights(dataset):
    """Assign unit sampling weight to every training clip."""
    return torch.ones(len(dataset)).float()


class MixupDataset(Dataset):
    """Mix two waveforms and their strong targets with a shared coefficient."""

    def __init__(self, dataset, beta=2, rate=0.5):
        self.beta = beta
        self.rate = rate
        self.dataset = dataset
        print(f"Mixing up waveforms from dataset of len {len(dataset)}")

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            batch1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            batch2 = self.dataset[idx2]
            x1, x2 = batch1["audio"], batch2["audio"]
            y1, y2 = batch1["strong"], batch2["strong"]
            coefficient = np.random.beta(self.beta, self.beta)
            coefficient = max(coefficient, 1.0 - coefficient)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = x1 * coefficient + x2 * (1.0 - coefficient)
            x = x - x.mean()
            batch1["audio"] = x
            batch1["strong"] = y1 * coefficient + y2 * (1.0 - coefficient)
            return batch1
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self,
        sampler,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle, seed=seed
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


def get_weighted_sampler(
    samples_weights,
    epoch_len=100_000,
    sampler_replace=False,
    seed=0,
):
    num_nodes = int(os.environ.get("WORLD_SIZE", 1))
    ddp = int(os.environ.get("DDP", 1))
    num_nodes = max(ddp, num_nodes)
    rank = int(os.environ.get("NODE_RANK", 0))
    return DistributedSamplerWrapper(
        sampler=WeightedRandomSampler(
            samples_weights, num_samples=epoch_len, replacement=sampler_replace
        ),
        dataset=range(epoch_len),
        num_replicas=num_nodes,
        rank=rank,
        seed=seed,
    )
