"""Event-based sound event detection metrics backed by ``sed_eval``."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import sed_eval
from dcase_util.containers import MetaDataContainer

GroundTruthEvent = Tuple[float, float, str]


def _normalized_filename(filename: str) -> str:
    """Return the filename format used by both sed_eval event containers."""
    return f"{Path(str(filename)).stem}.wav"


def _finite_or_zero(value: float) -> float:
    value = float(value)
    return value if isfinite(value) else 0.0


def compute_macro_event_metrics(
    predictions: pd.DataFrame,
    ground_truth: Mapping[str, Sequence[GroundTruthEvent]],
    event_classes: Sequence[str],
    onset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
) -> Dict[str, float]:
    """Compute macro event precision, recall and F1 over a complete dataset.

    Events are matched one-to-one by ``sed_eval``. An onset is correct within
    ``onset_collar`` seconds. The offset tolerance is the larger of that
    absolute collar and ``offset_collar_rate`` times the reference duration.

    Undefined class-wise precision/F1 values (for example, no predicted event
    for a referenced class) are treated as zero before macro averaging.
    """
    classes = list(dict.fromkeys(event_classes))
    if not classes:
        raise ValueError("event_classes must contain at least one class")
    if onset_collar < 0:
        raise ValueError("onset_collar must be non-negative")
    if offset_collar_rate < 0:
        raise ValueError("offset_collar_rate must be non-negative")

    required_columns = {"filename", "event_label", "onset", "offset"}
    if predictions.empty:
        predictions = pd.DataFrame(columns=sorted(required_columns))
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Event predictions are missing columns: {sorted(missing_columns)}"
        )

    class_set = set(classes)
    evaluated_filenames = {_normalized_filename(file_id) for file_id in ground_truth}
    prediction_events_by_file = {}
    predicted_classes = set()
    prediction_event_count = 0
    for row in predictions.itertuples(index=False):
        event_label = str(row.event_label)
        onset = float(row.onset)
        offset = float(row.offset)
        if event_label not in class_set or offset <= onset:
            continue
        filename = _normalized_filename(row.filename)
        if filename not in evaluated_filenames:
            continue
        prediction_events_by_file.setdefault(filename, []).append(
            {
                "file": filename,
                "event_label": event_label,
                "event_onset": onset,
                "event_offset": offset,
            }
        )
        predicted_classes.add(event_label)
        prediction_event_count += 1

    metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        evaluate_onset=True,
        evaluate_offset=True,
        t_collar=float(onset_collar),
        percentage_of_length=float(offset_collar_rate),
        event_matching_type="optimal",
    )

    reference_event_count = 0
    referenced_classes = set()
    for file_id, events in ground_truth.items():
        filename = _normalized_filename(file_id)
        reference_events = [
            {
                "file": filename,
                "event_label": str(event_label),
                "event_onset": float(onset),
                "event_offset": float(offset),
            }
            for onset, offset, event_label in events
            if event_label in class_set and float(offset) > float(onset)
        ]
        reference_event_count += len(reference_events)
        referenced_classes.update(event["event_label"] for event in reference_events)
        metric.evaluate(
            reference_event_list=MetaDataContainer(reference_events),
            estimated_event_list=MetaDataContainer(
                prediction_events_by_file.get(filename, [])
            ),
        )

    class_wise = metric.results_class_wise_metrics()
    overall = metric.results_overall_metrics()["f_measure"]
    precision = []
    recall = []
    f1 = []
    for event_class in classes:
        f_measure = class_wise[event_class]["f_measure"]
        precision.append(_finite_or_zero(f_measure["precision"]))
        recall.append(_finite_or_zero(f_measure["recall"]))
        f1.append(_finite_or_zero(f_measure["f_measure"]))

    macro_f1 = float(np.mean(f1))
    return {
        "event_precision": float(np.mean(precision)),
        "event_recall": float(np.mean(recall)),
        "event_f1": macro_f1,
        "event_precision_micro": _finite_or_zero(overall["precision"]),
        "event_recall_micro": _finite_or_zero(overall["recall"]),
        "event_f1_micro": _finite_or_zero(overall["f_measure"]),
        "evaluated_class_count": float(len(classes)),
        "referenced_class_count": float(len(referenced_classes)),
        "predicted_class_count": float(len(predicted_classes)),
        "nonzero_f1_class_count": float(sum(value > 0.0 for value in f1)),
        "reference_event_count": float(reference_event_count),
        "prediction_event_count": float(prediction_event_count),
    }


def main():
    """Evaluate a saved AEDR-SED checkpoint on the configured dataset."""
    from train import main as dispatch

    dispatch(evaluate=True)


if __name__ == "__main__":
    main()
