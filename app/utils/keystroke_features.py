from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .keystroke_capture import KeystrokeEvent


def _pair_events(events: Iterable[KeystrokeEvent]) -> List[Dict[str, float]]:
    holds = []
    down_stack = []
    for event in events:
        if event.event == "down":
            down_stack.append(event)
        elif event.event == "up" and down_stack:
            start = down_stack.pop(0)
            holds.append({"key": start.key, "start": start.ts, "end": event.ts})
    return holds


def build_feature_vector(events: List[KeystrokeEvent]) -> np.ndarray:
    """Convert raw keystroke events into a numeric feature vector."""
    if not events:
        return np.zeros(9, dtype=float)

    holds = _pair_events(events)
    hold_durations = [h["end"] - h["start"] for h in holds if h["end"] >= h["start"]]
    hold_durations = hold_durations or [0.0]

    down_times = [e.ts for e in events if e.event == "down"]
    up_times = [e.ts for e in events if e.event == "up"]
    down_times.sort()
    up_times.sort()

    # Flight time: gap between consecutive key presses
    flight_times = [
        down_times[i + 1] - down_times[i]
        for i in range(len(down_times) - 1)
        if down_times[i + 1] >= down_times[i]
    ] or [0.0]

    # Inter-key delay: release of key i to press of key i+1
    inter_key_delays = []
    for idx in range(min(len(up_times), len(down_times) - 1)):
        delay = down_times[idx + 1] - up_times[idx]
        if delay >= 0:
            inter_key_delays.append(delay)
    if not inter_key_delays:
        inter_key_delays = [0.0]

    total_duration = (max(up_times or down_times) - min(down_times)) if down_times else 0.0

    def stats(values: List[float]) -> tuple:
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    hold_mean, hold_std, hold_min, hold_max = stats(hold_durations)
    flight_mean, flight_std, _, flight_max = stats(flight_times)
    inter_mean, inter_std, _, inter_max = stats(inter_key_delays)

    return np.array(
        [
            hold_mean,
            hold_std,
            hold_min,
            hold_max,
            flight_mean,
            flight_std,
            flight_max,
            inter_mean,
            inter_std,
            total_duration,
        ],
        dtype=float,
    )


def stack_samples(samples: List[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.empty((0, 10))
    return np.vstack(samples)
