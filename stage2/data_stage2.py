from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from config_stage2 import (
    CLIP_LEN,
    EXPECTED_NO_STAGE2_READY_SUBJECTS,
    EXPECTED_STAGE2_HOLD_RECORDINGS,
    EXPECTED_STAGE2_QC_MANIFEST_SHA256,
    EXPECTED_STAGE2_READY_RECORDINGS,
    EXPECTED_STAGE2_READY_SUBJECTS,
    EXPECTED_STAGE2_READY_SUBJECT_STATES,
    ROI_SIZE,
    SCALE_TO_01,
    SEED,
    STAGE2_QC_MANIFEST,
    VAL_CLIPS_PER_RECORDING,
    VAL_CLIP_QUANTILES,
    VAL_RECORDINGS_PER_SUBJECT,
)
from io_utils import sha256_file


@dataclass(frozen=True)
class RecordingRef:
    h5_path: str
    subject_id: str
    state: str
    camera: str
    view: str
    fallback_ratio: float
    n_clips: int
    n_usable: int


def _seed_from_parts(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def load_stage2_manifest(
    manifest_path: Path = STAGE2_QC_MANIFEST,
    verify_frozen_hash: bool = True,
):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    digest = sha256_file(manifest_path)
    if verify_frozen_hash and digest != EXPECTED_STAGE2_QC_MANIFEST_SHA256:
        raise RuntimeError(
            "Stage-2 QC manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_STAGE2_QC_MANIFEST_SHA256}\n"
            f"Actual:   {digest}\n"
            "Training is blocked until the changed manifest is re-audited."
        )

    ready: List[RecordingRef] = []
    hold: List[dict] = []
    rows_total = 0

    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject_id", "state", "camera", "h5_path",
            "n_usable", "n_clips", "fallback_ratio", "status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"Stage-2 manifest missing required columns: {sorted(missing)}"
            )

        for row in reader:
            rows_total += 1
            if row["status"].strip().upper() != "READY":
                hold.append(dict(row))
                continue

            path = row["h5_path"].strip()
            ref = RecordingRef(
                h5_path=path,
                subject_id=str(row["subject_id"]).strip(),
                state=str(row["state"]).strip(),
                camera=str(row["camera"]).strip(),
                view=str(row.get("view", "")).strip(),
                fallback_ratio=float(row["fallback_ratio"]),
                n_clips=int(float(row["n_clips"])),
                n_usable=int(float(row["n_usable"])),
            )

            if ref.n_clips <= 0:
                raise RuntimeError(f"READY Stage-2 row has no clips: {path}")
            if not Path(path).exists():
                raise FileNotFoundError(f"READY Stage-2 H5 missing: {path}")

            ready.append(ref)

    subjects = {r.subject_id for r in ready}
    subject_states = {(r.subject_id, r.state) for r in ready}

    if rows_total != EXPECTED_STAGE2_READY_RECORDINGS + EXPECTED_STAGE2_HOLD_RECORDINGS:
        raise RuntimeError(
            f"Expected 3600 Stage-2 manifest rows, found {rows_total}."
        )
    if len(ready) != EXPECTED_STAGE2_READY_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE2_READY_RECORDINGS} READY recordings, "
            f"found {len(ready)}."
        )
    if len(hold) != EXPECTED_STAGE2_HOLD_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE2_HOLD_RECORDINGS} HOLD recordings, "
            f"found {len(hold)}."
        )
    if len(subjects) != EXPECTED_STAGE2_READY_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE2_READY_SUBJECTS} READY subjects, "
            f"found {len(subjects)}."
        )
    if len(subject_states) != EXPECTED_STAGE2_READY_SUBJECT_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE2_READY_SUBJECT_STATES} READY subject-states, "
            f"found {len(subject_states)}."
        )

    summary = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "rows_total": rows_total,
        "ready_recordings": len(ready),
        "hold_recordings": len(hold),
        "ready_subjects": len(subjects),
        "ready_subject_states": len(subject_states),
    }
    return ready, hold, summary


def recordings_for_subjects(
    recordings: Sequence[RecordingRef],
    subjects: Sequence[str],
) -> List[RecordingRef]:
    keep = set(map(str, subjects))
    return [r for r in recordings if r.subject_id in keep]


def group_indices_by_subject(
    recordings: Sequence[RecordingRef],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, r in enumerate(recordings):
        out.setdefault(r.subject_id, []).append(i)
    return out


class Stage2RPPGDataset(Dataset):
    """
    Tuple index = (recording_index, logical_clip_index).

    No Stage-2 image augmentation is used. The target is the authoritative PPG
    copied into H5 from ppg_sync by the compact extractor, z-normalized per
    128-frame clip.
    """

    def __init__(
        self,
        recordings: Sequence[RecordingRef],
        clip_len: int = CLIP_LEN,
        scale_to_01: bool = SCALE_TO_01,
    ):
        self.recordings = list(recordings)
        self.clip_len = int(clip_len)
        self.scale_to_01 = bool(scale_to_01)

    def __len__(self):
        # DataLoader uses custom batch samplers; this is only a structural length.
        return len(self.recordings)

    def __getitem__(self, index):
        if not isinstance(index, (tuple, list)) or len(index) != 2:
            raise RuntimeError(
                "Stage2RPPGDataset requires (recording_index, clip_index)."
            )

        ri, ci = int(index[0]), int(index[1])
        ref = self.recordings[ri]

        with h5py.File(ref.h5_path, "r") as h5:
            starts = h5["clip_starts"]
            if ci < 0 or ci >= len(starts):
                raise IndexError((ri, ci))

            start = int(starts[ci])
            end = start + self.clip_len

            roi = h5["roi"][start:end]
            ppg = h5["ppg"][start:end].astype(np.float32)
            fps = float(
                h5.attrs.get(
                    "meta_fps",
                    h5.attrs.get("video_fps_reported", 30.0),
                )
            )

        if roi.shape != (self.clip_len, ROI_SIZE, ROI_SIZE, 3):
            raise RuntimeError(
                f"Bad Stage-2 ROI slice: {ref.h5_path}, start={start}, shape={roi.shape}"
            )
        if ppg.shape != (self.clip_len,) or not np.isfinite(ppg).all():
            raise RuntimeError(
                f"Bad Stage-2 PPG slice: {ref.h5_path}, start={start}"
            )

        ppg_mean = float(ppg.mean())
        ppg_std = float(ppg.std())
        if not np.isfinite(ppg_std) or ppg_std < 1e-6:
            raise RuntimeError(
                f"Degenerate PPG target: {ref.h5_path}, start={start}, std={ppg_std}"
            )

        x = roi.transpose(3, 0, 1, 2).astype(np.float32, copy=False)
        if self.scale_to_01:
            x /= 255.0

        y = (ppg - ppg_mean) / (ppg_std + 1e-8)

        meta = {
            "subject_id": ref.subject_id,
            "state": ref.state,
            "camera": ref.camera,
            "view": ref.view,
            "recording_index": ri,
            "clip_index": ci,
            "start_idx": start,
            "fps": fps,
            "h5_path": ref.h5_path,
            "target_raw_mean": ppg_mean,
            "target_raw_std": ppg_std,
        }

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(y.astype(np.float32, copy=False)),
            meta,
        )


class SubjectBalancedClipBatchSampler(Sampler[List[Tuple[int, int]]]):
    """
    Equal subject exposure with no repeated subject inside a training batch.

    Each subject draw:
      1) chooses one READY recording uniformly for that subject,
      2) chooses one logical clip uniformly in that recording.
    """

    def __init__(
        self,
        recordings: Sequence[RecordingRef],
        batch_size: int,
        seed: int = SEED,
    ):
        self.recordings = list(recordings)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.by_subject = group_indices_by_subject(self.recordings)
        self.subjects = sorted(self.by_subject)

        if self.batch_size < 2:
            raise ValueError("Stage-2 batch size must be >=2.")
        if len(self.subjects) < self.batch_size:
            raise ValueError("Not enough subjects for unique-subject batches.")

        target = max(len(self.recordings), self.batch_size)
        self.total_samples = (target // self.batch_size) * self.batch_size
        self.n_batches = self.total_samples // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        rng = random.Random(
            _seed_from_parts(self.seed, "stage2_train", self.epoch)
        )

        n_subjects = len(self.subjects)
        base = self.total_samples // n_subjects
        rem = self.total_samples % n_subjects

        shuffled = self.subjects.copy()
        rng.shuffle(shuffled)

        quotas = {s: base for s in self.subjects}
        for s in shuffled[:rem]:
            quotas[s] += 1

        for _batch_i in range(self.n_batches):
            eligible = [s for s, q in quotas.items() if q > 0]
            if len(eligible) < self.batch_size:
                raise RuntimeError(
                    "Unable to construct a unique-subject Stage-2 batch."
                )

            rng.shuffle(eligible)
            eligible.sort(key=lambda s: quotas[s], reverse=True)
            chosen = eligible[: self.batch_size]

            batch = []
            for subject in chosen:
                ri = rng.choice(self.by_subject[subject])
                ref = self.recordings[ri]
                ci = rng.randrange(ref.n_clips)
                batch.append((ri, ci))
                quotas[subject] -= 1

            yield batch


def _deterministic_pick(options: List[RecordingRef], *parts) -> RecordingRef:
    options = sorted(
        options,
        key=lambda r: (
            r.fallback_ratio,
            r.camera,
            r.h5_path,
        ),
    )
    idx = _seed_from_parts(SEED, *parts) % len(options)
    return options[idx]


def select_validation_recordings(
    recordings: Sequence[RecordingRef],
    subject_ids: Sequence[str],
) -> List[RecordingRef]:
    wanted = set(map(str, subject_ids))
    by_subject: Dict[str, List[RecordingRef]] = {}

    for r in recordings:
        if r.subject_id in wanted:
            by_subject.setdefault(r.subject_id, []).append(r)

    selected: List[RecordingRef] = []
    for subject in sorted(wanted, key=lambda x: int(x) if x.isdigit() else x):
        options = by_subject.get(subject, [])
        if not options:
            raise RuntimeError(
                f"Development subject {subject} has no Stage-2 READY recording."
            )

        before = [r for r in options if r.state == "before"]
        after = [r for r in options if r.state == "after"]
        chosen: List[RecordingRef] = []

        if before:
            chosen.append(
                _deterministic_pick(before, "val_recording", subject, "before")
            )
        if after and len(chosen) < VAL_RECORDINGS_PER_SUBJECT:
            chosen.append(
                _deterministic_pick(after, "val_recording", subject, "after")
            )

        if len(chosen) < VAL_RECORDINGS_PER_SUBJECT:
            remaining = [r for r in options if r not in chosen]
            remaining = sorted(
                remaining,
                key=lambda r: (
                    _seed_from_parts(
                        SEED, "val_fallback", subject, r.camera, r.state, r.h5_path
                    ),
                    r.h5_path,
                ),
            )
            for r in remaining:
                chosen.append(r)
                if len(chosen) >= VAL_RECORDINGS_PER_SUBJECT:
                    break

        selected.extend(chosen)

    return selected


def validation_clip_indices(n_clips: int) -> List[int]:
    if n_clips <= 0:
        return []
    idxs = []
    for q in VAL_CLIP_QUANTILES:
        idx = int(round(float(q) * (n_clips - 1)))
        idx = min(max(idx, 0), n_clips - 1)
        if idx not in idxs:
            idxs.append(idx)
    return idxs[:VAL_CLIPS_PER_RECORDING]


class FixedValidationBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(
        self,
        selected_recordings: Sequence[RecordingRef],
        batch_size: int,
    ):
        self.recordings = list(selected_recordings)
        self.batch_size = int(batch_size)

        items: List[Tuple[int, int]] = []
        for ri, ref in enumerate(self.recordings):
            for ci in validation_clip_indices(ref.n_clips):
                items.append((ri, ci))

        self.items = items
        self.batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        yield from self.batches


def full_h5_ppg_audit(recordings: Sequence[RecordingRef], clip_len=CLIP_LEN):
    """
    Deterministic full audit of all Stage-2 READY H5 targets.

    It verifies structural alignment and computes every logical clip's PPG
    standard deviation using prefix sums without materializing all clips.
    """
    minimum_std = float("inf")
    minimum_recording = None
    checked_clips = 0
    source_attr_missing = 0

    for ref in recordings:
        with h5py.File(ref.h5_path, "r") as h5:
            for key in ("roi", "ppg", "ppg_sync_dt", "clip_starts"):
                if key not in h5:
                    raise RuntimeError(f"{ref.h5_path} missing dataset {key}")

            n_roi = int(h5["roi"].shape[0])
            n_ppg = int(h5["ppg"].shape[0])
            n_dt = int(h5["ppg_sync_dt"].shape[0])
            starts = h5["clip_starts"][:].astype(np.int64)
            ppg = h5["ppg"][:].astype(np.float64)
            dt = h5["ppg_sync_dt"][:].astype(np.float64)

            if not (n_roi == n_ppg == n_dt == ref.n_usable):
                raise RuntimeError(
                    f"Stage-2 alignment mismatch: {ref.h5_path} "
                    f"roi={n_roi} ppg={n_ppg} dt={n_dt} manifest={ref.n_usable}"
                )
            if len(starts) != ref.n_clips:
                raise RuntimeError(
                    f"clip_starts mismatch: {ref.h5_path} "
                    f"h5={len(starts)} manifest={ref.n_clips}"
                )
            if not np.isfinite(ppg).all() or not np.isfinite(dt).all():
                raise RuntimeError(f"Non-finite PPG/dt in {ref.h5_path}")
            if len(starts) and int(starts.max()) + clip_len > n_ppg:
                raise RuntimeError(f"Out-of-range clip_start in {ref.h5_path}")

            if "source_ppg_sync" not in h5.attrs:
                source_attr_missing += 1

            # Prefix-sum variance for every logical clip.
            cs = np.concatenate([[0.0], np.cumsum(ppg)])
            cs2 = np.concatenate([[0.0], np.cumsum(ppg * ppg)])
            ends = starts + clip_len
            sums = cs[ends] - cs[starts]
            sums2 = cs2[ends] - cs2[starts]
            means = sums / clip_len
            vars_ = np.maximum(sums2 / clip_len - means * means, 0.0)
            stds = np.sqrt(vars_)

            if len(stds):
                local_i = int(np.argmin(stds))
                local_min = float(stds[local_i])
                if local_min < minimum_std:
                    minimum_std = local_min
                    minimum_recording = {
                        "h5_path": ref.h5_path,
                        "clip_index": local_i,
                        "start_idx": int(starts[local_i]),
                    }
                checked_clips += len(stds)

    return {
        "recordings_checked": len(recordings),
        "logical_clips_checked": checked_clips,
        "minimum_segment_ppg_std": minimum_std,
        "minimum_segment_location": minimum_recording,
        "source_ppg_sync_attr_missing_count": source_attr_missing,
    }
