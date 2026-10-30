from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
from config_stage2 import BEST_LAG_MAX_SECONDS


def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _corr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def waveform_metrics_per_sample(pred, target, fps=None):
    p, y = _np(pred), _np(target)
    fs = None if fps is None else _np(fps).reshape(-1)

    keys = [
        "mae", "rmse", "pearson_r", "abs_pearson_r", "cosine",
        "derivative_rmse", "pred_std", "target_std", "std_ratio",
        "best_lag_r", "best_lag_frames", "best_lag_seconds",
    ]
    vals = {k: [] for k in keys}

    for i, (a, b) in enumerate(zip(p, y)):
        vals["mae"].append(float(np.mean(np.abs(a - b))))
        vals["rmse"].append(float(np.sqrt(np.mean((a - b) ** 2))))
        r0 = _corr(a, b)
        vals["pearson_r"].append(r0)
        vals["abs_pearson_r"].append(abs(r0))
        vals["cosine"].append(float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        ))
        vals["derivative_rmse"].append(float(np.sqrt(np.mean(
            (np.diff(a) - np.diff(b)) ** 2
        ))))

        ps = float(np.std(a))
        ts = float(np.std(b))
        vals["pred_std"].append(ps)
        vals["target_std"].append(ts)
        vals["std_ratio"].append(ps / (ts + 1e-12))

        f = float(fs[i]) if fs is not None else 30.0
        max_lag = max(1, min(len(a)//3, int(round(BEST_LAG_MAX_SECONDS * f))))
        best_r, best_lag = -1.0, 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                aa, bb = a[:lag], b[-lag:]
            elif lag > 0:
                aa, bb = a[lag:], b[:-lag]
            else:
                aa, bb = a, b
            r = _corr(aa, bb)
            if r > best_r:
                best_r, best_lag = r, lag
        vals["best_lag_r"].append(best_r)
        vals["best_lag_frames"].append(float(best_lag))
        vals["best_lag_seconds"].append(float(best_lag / max(f, 1e-6)))

    return {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()}


def spectral_correlation_per_sample(pred, target, fps, pulse_band=(0.5, 4.0), nfft=512):
    p, y = _np(pred), _np(target)
    fs = _np(fps).reshape(-1)
    out = []
    for a, b, f in zip(p, y, fs):
        if not np.isfinite(f) or f <= 0:
            out.append(np.nan); continue
        A = np.abs(np.fft.rfft(a - a.mean(), n=nfft)) ** 2
        B = np.abs(np.fft.rfft(b - b.mean(), n=nfft)) ** 2
        hz = np.fft.rfftfreq(nfft, d=1.0 / float(f))
        mask = (hz >= pulse_band[0]) & (hz <= pulse_band[1])
        A, B = A[mask], B[mask]
        if len(A) < 3 or np.std(A) < 1e-12 or np.std(B) < 1e-12:
            r = 0.0
        else:
            A = A / (A.sum() + 1e-12)
            B = B / (B.sum() + 1e-12)
            r = float(np.corrcoef(A, B)[0, 1])
        out.append(r)
    return np.asarray(out, dtype=np.float64)


class SubjectMetricAccumulator:
    def __init__(self):
        self.by_subject = defaultdict(lambda: defaultdict(list))
        self.clip_values = defaultdict(list)
        self.by_camera_subject = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.by_state_subject = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.recordings = set()
        self.n_samples = 0

    def add(self, subject_ids, h5_paths, metrics, cameras=None, states=None):
        subject_ids = [str(x) for x in subject_ids]
        h5_paths = [str(x) for x in h5_paths]
        cameras = ["unknown"] * len(subject_ids) if cameras is None else [str(x) for x in cameras]
        states = ["unknown"] * len(subject_ids) if states is None else [str(x) for x in states]
        n = len(subject_ids)

        for key, arr in metrics.items():
            if len(np.asarray(arr).reshape(-1)) != n:
                raise RuntimeError(f"Metric {key} length mismatch")

        for i, subject in enumerate(subject_ids):
            self.recordings.add(h5_paths[i])
            self.n_samples += 1
            cam, state = cameras[i], states[i]
            for key, arr in metrics.items():
                value = float(np.asarray(arr).reshape(-1)[i])
                if np.isfinite(value):
                    self.by_subject[subject][key].append(value)
                    self.clip_values[key].append(value)
                    self.by_camera_subject[cam][subject][key].append(value)
                    self.by_state_subject[state][subject][key].append(value)

    @staticmethod
    def _subject_balanced(group):
        keys = sorted({k for s in group.values() for k in s})
        out = {}
        for key in keys:
            means = [
                float(np.mean(v[key])) for v in group.values()
                if key in v and len(v[key])
            ]
            out[key] = float(np.mean(means)) if means else float("nan")
        out["n_subjects"] = len(group)
        return out

    def finalize(self):
        result = self._subject_balanced(self.by_subject)
        for key, values in self.clip_values.items():
            result[f"clip_{key}"] = float(np.mean(values)) if values else float("nan")
        result["n_recordings"] = len(self.recordings)
        result["n_samples"] = self.n_samples
        return result

    def grouped_finalize(self):
        return {
            "camera": {
                cam: self._subject_balanced(group)
                for cam, group in sorted(self.by_camera_subject.items())
            },
            "state": {
                state: self._subject_balanced(group)
                for state, group in sorted(self.by_state_subject.items())
            },
        }
