"""Strong-label encoding, event decoding and data-loader worker seeding.

Encoding utilities are adapted from https://github.com/DCASE-REPO/DESED_task
and https://github.com/DCASE-REPO/dcase_util.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.special
import torch
from dcase_util.data import DecisionEncoder
from sed_scores_eval.base_modules.scores import create_score_dataframe


def batched_decode_preds(
    strong_preds,
    filenames,
    encoder,
    thresholds=None,
    median_filter=None,
):
    """Return raw scores, median-filtered scores and thresholded event tables."""
    thresholds = [0.5] if thresholds is None else thresholds
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {threshold: pd.DataFrame() for threshold in thresholds}
    for index in range(strong_preds.shape[0]):
        audio_id = Path(filenames[index]).stem
        scores = strong_preds[index].transpose(0, 1).detach().cpu().numpy()
        timestamps = encoder._frame_to_time(np.arange(len(scores) + 1))
        scores_raw[audio_id] = create_score_dataframe(
            scores=scores,
            timestamps=timestamps,
            event_classes=encoder.labels,
        )
        if median_filter is not None:
            scores = scipy.ndimage.median_filter(scores, (median_filter, 1))
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=scores,
            timestamps=timestamps,
            event_classes=encoder.labels,
        )
        for threshold in thresholds:
            events = pd.DataFrame(
                encoder.decode_strong(scores > threshold),
                columns=["event_label", "onset", "offset"],
            )
            events["filename"] = f"{audio_id}.wav"
            prediction_dfs[threshold] = pd.concat(
                [prediction_dfs[threshold], events],
                ignore_index=True,
            )
    return scores_raw, scores_postprocessed, prediction_dfs


def decode_score_dataframes_to_events(
    score_dataframes,
    encoder,
    threshold,
    scores_are_logits=False,
):
    """Decode cached frame scores without repeating median filtering."""
    predictions = []
    for audio_id, dataframe in score_dataframes.items():
        scores = dataframe[encoder.labels].to_numpy(copy=False)
        if scores_are_logits:
            scores = scipy.special.expit(scores)
        events = pd.DataFrame(
            encoder.decode_strong(scores > float(threshold)),
            columns=["event_label", "onset", "offset"],
        )
        events["filename"] = f"{Path(audio_id).stem}.wav"
        predictions.append(events)
    if not predictions:
        return pd.DataFrame(columns=["event_label", "onset", "offset", "filename"])
    return pd.concat(predictions, ignore_index=True)


def worker_init_fn(worker_id):
    """Seed NumPy, Python and PyTorch independently for each loader worker."""
    seed = (torch.initial_seed() + worker_id * 1000) % 2**31
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class ManyHotEncoder:
    """Rasterize event intervals and decode contiguous active frame regions."""

    def __init__(self, labels, audio_len=10, frame_hop=160, net_pooling=4, fs=16000):
        self.labels = list(labels)
        self.audio_len = audio_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling
        self.n_frames = int(int(audio_len * fs / frame_hop) / net_pooling)

    def _time_to_frame(self, time):
        frame = time * self.fs / self.frame_hop / self.net_pooling
        return np.clip(frame, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling / (self.fs / self.frame_hop)
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events):
        """Encode onset/offset times in seconds as a [frames, classes] array."""
        targets = np.zeros((self.n_frames, len(self.labels)))
        for _, event in events.iterrows():
            if pd.isna(event["event_label"]):
                continue
            class_index = self.labels.index(event["event_label"])
            onset = int(self._time_to_frame(event["onset"]))
            offset = int(np.ceil(self._time_to_frame(event["offset"])))
            confidence = event["confidence"] if "confidence" in events.columns else 1
            targets[onset:offset, class_index] = confidence
        return targets

    def decode_strong(self, labels):
        """Decode a [frames, classes] binary array to label/onset/offset rows."""
        events = []
        decoder = DecisionEncoder()
        for index, activity in enumerate(labels.T):
            for onset, offset in decoder.find_contiguous_regions(activity):
                events.append(
                    [
                        self.labels[index],
                        self._frame_to_time(onset),
                        self._frame_to_time(offset),
                    ]
                )
        return events
