from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from config_stage1 import (
    BRIGHTNESS_GAIN,
    CLIP_LEN,
    CONTRAST_GAIN,
    EXPECTED_STAGE1_QC_MANIFEST_SHA256,
    EXPECTED_STAGE1_READY_RECORDINGS,
    EXPECTED_STAGE1_SUBJECT_STATES,
    EXPECTED_TOTAL_SUBJECTS,
    GAUSSIAN_NOISE_STD,
    HFLIP_PROB,
    ROI_SIZE,
    SCALE_TO_01,
    SEED,
    STAGE1_QC_MANIFEST,
)
from io_utils import sha256_file


CAMERA_ORDER = {
    "FullHDwebcam": 0,
    "USBVideo": 1,
    "IriunWebcam": 2,
}
STATE_ORDER = {"before": 0, "after": 1}


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


def _to_bool(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def load_stage1_manifest(
    manifest_path: Path = STAGE1_QC_MANIFEST,
    verify_frozen_hash: bool = True,
) -> Tuple[List[RecordingRef], List[dict], dict]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    digest = sha256_file(manifest_path)
    if verify_frozen_hash and digest != EXPECTED_STAGE1_QC_MANIFEST_SHA256:
        raise RuntimeError(
            "Stage-1 QC manifest SHA256 mismatch.\n"
            f"Expected: {EXPECTED_STAGE1_QC_MANIFEST_SHA256}\n"
            f"Actual:   {digest}\n"
            "Do not train until the changed manifest is deliberately re-audited and frozen."
        )

    accepted: List[RecordingRef] = []
    rejected: List[dict] = []
    all_rows = 0

    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject_id", "state", "camera", "view", "h5_path",
            "n_usable", "n_clips", "fallback_ratio", "status", "hold_reasons",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Stage-1 manifest missing columns: {sorted(missing)}")

        for row in reader:
            all_rows += 1
            if row["status"].strip().upper() != "READY":
                rejected.append(dict(row))
                continue

            ref = RecordingRef(
                h5_path=row["h5_path"].strip(),
                subject_id=str(row["subject_id"]).strip(),
                state=row["state"].strip(),
                camera=row["camera"].strip(),
                view=row["view"].strip(),
                fallback_ratio=float(row["fallback_ratio"]),
                n_clips=int(float(row["n_clips"])),
                n_usable=int(float(row["n_usable"])),
            )
            if ref.n_clips <= 0:
                raise RuntimeError(f"READY recording has no clips: {ref.h5_path}")
            if not Path(ref.h5_path).exists():
                raise FileNotFoundError(f"Manifest READY H5 does not exist: {ref.h5_path}")
            accepted.append(ref)

    subjects = {r.subject_id for r in accepted}
    subject_states = {(r.subject_id, r.state) for r in accepted}

    summary = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "rows_total": all_rows,
        "ready_recordings": len(accepted),
        "hold_recordings": len(rejected),
        "ready_subjects": len(subjects),
        "ready_subject_states": len(subject_states),
    }

    if len(accepted) != EXPECTED_STAGE1_READY_RECORDINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE1_READY_RECORDINGS} Stage-1 READY recordings, "
            f"found {len(accepted)}."
        )
    if len(subjects) != EXPECTED_TOTAL_SUBJECTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SUBJECTS} Stage-1 subjects, found {len(subjects)}."
        )
    if len(subject_states) != EXPECTED_STAGE1_SUBJECT_STATES:
        raise RuntimeError(
            f"Expected {EXPECTED_STAGE1_SUBJECT_STATES} subject-state groups, "
            f"found {len(subject_states)}."
        )

    return accepted, rejected, summary


def recordings_for_subjects(
    recordings: Sequence[RecordingRef],
    subjects: Sequence[str],
) -> List[RecordingRef]:
    keep = set(map(str, subjects))
    return [r for r in recordings if r.subject_id in keep]


def group_recording_indices_by_subject(
    recordings: Sequence[RecordingRef],
) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, r in enumerate(recordings):
        out.setdefault(r.subject_id, []).append(i)
    return out


class Stage1MCDH5Dataset(Dataset):
    """
    One dataset item is one selected logical 128-frame clip from one READY recording.

    Training index can be `(recording_index, draw_seed)`. This makes clip selection
    and augmentation stateless/reproducible across DataLoader workers.
    """

    def __init__(
        self,
        recordings: Sequence[RecordingRef],
        clip_len: int = CLIP_LEN,
        training: bool = True,
        deterministic_validation_pair: bool = True,
        hflip_prob: float = HFLIP_PROB,
        brightness_gain: float = BRIGHTNESS_GAIN,
        contrast_gain: float = CONTRAST_GAIN,
        gaussian_noise_std: float = GAUSSIAN_NOISE_STD,
        scale_to_01: bool = SCALE_TO_01,
    ):
        self.recordings = list(recordings)
        self.clip_len = int(clip_len)
        self.training = bool(training)
        self.deterministic_validation_pair = bool(deterministic_validation_pair)
        self.hflip_prob = float(hflip_prob)
        self.brightness_gain = float(brightness_gain)
        self.contrast_gain = float(contrast_gain)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.scale_to_01 = bool(scale_to_01)

    def __len__(self):
        return len(self.recordings)

    def _parse_index(self, index):
        if isinstance(index, (tuple, list)) and len(index) == 2:
            return int(index[0]), int(index[1])
        return int(index), None

    def __getitem__(self, index):
        idx, draw_seed = self._parse_index(index)
        ref = self.recordings[idx]

        if self.training:
            if draw_seed is None:
                raise RuntimeError(
                    "Training dataset requires seeded sampler indices for reproducibility."
                )
            py_rng = random.Random(draw_seed)
            np_rng = np.random.default_rng(draw_seed)
        else:
            py_rng = random.Random(SEED)
            np_rng = np.random.default_rng(SEED)

        with h5py.File(ref.h5_path, "r") as h5:
            starts = h5["clip_starts"][:]
            if len(starts) == 0:
                raise RuntimeError(f"No clip_starts: {ref.h5_path}")

            if self.training:
                start = int(starts[py_rng.randrange(len(starts))])
            else:
                # A fixed central logical clip makes validation deterministic.
                start = int(starts[len(starts) // 2])

            end = start + self.clip_len
            roi = h5["roi"][start:end]

        if roi.shape != (self.clip_len, ROI_SIZE, ROI_SIZE, 3):
            raise RuntimeError(
                f"Bad ROI clip slice {ref.h5_path}: start={start}, shape={roi.shape}"
            )

        clip = roi.transpose(3, 0, 1, 2).astype(np.float32, copy=False)
        if self.scale_to_01:
            clip /= 255.0

        if self.training:
            v1 = self._aug_random(clip.copy(), py_rng, np_rng)
            v2 = self._aug_random(clip.copy(), py_rng, np_rng)
        elif self.deterministic_validation_pair:
            v1 = clip.copy()
            v2 = self._aug_deterministic(clip.copy())
        else:
            v1 = clip.copy()
            v2 = clip.copy()

        meta = {
            "subject_id": ref.subject_id,
            "state": ref.state,
            "camera": ref.camera,
            "view": ref.view,
            "start_idx": start,
            "fallback_ratio": ref.fallback_ratio,
            "h5_path": ref.h5_path,
        }

        return (
            torch.from_numpy(np.ascontiguousarray(v1)),
            torch.from_numpy(np.ascontiguousarray(v2)),
            meta,
        )

    def _aug_random(self, clip, py_rng, np_rng):
        if py_rng.random() < self.hflip_prob:
            clip = clip[..., ::-1]

        if self.brightness_gain > 0:
            clip *= 1.0 + py_rng.uniform(
                -self.brightness_gain, self.brightness_gain
            )

        if self.contrast_gain > 0:
            factor = 1.0 + py_rng.uniform(
                -self.contrast_gain, self.contrast_gain
            )
            mean = clip.mean(axis=(1, 2, 3), keepdims=True)
            clip = (clip - mean) * factor + mean

        if self.gaussian_noise_std > 0:
            sigma = (
                self.gaussian_noise_std / 255.0
                if self.scale_to_01
                else self.gaussian_noise_std
            )
            clip += np_rng.normal(0, sigma, size=clip.shape).astype(np.float32)

        return np.clip(
            clip, 0, 1.0 if self.scale_to_01 else 255.0
        ).astype(np.float32, copy=False)

    def _aug_deterministic(self, clip):
        if self.brightness_gain > 0:
            clip *= 1.0 + self.brightness_gain * 0.5
        if self.contrast_gain > 0:
            factor = 1.0 - self.contrast_gain * 0.5
            mean = clip.mean(axis=(1, 2, 3), keepdims=True)
            clip = (clip - mean) * factor + mean
        return np.clip(
            clip, 0, 1.0 if self.scale_to_01 else 255.0
        ).astype(np.float32, copy=False)


def _seed_from_parts(*parts) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    # Keep seed in signed 31-bit range for broad RNG compatibility.
    return int(hashlib.sha256(text).hexdigest()[:8], 16) & 0x7FFFFFFF


class SubjectBalancedBatchSampler(Sampler[List[Tuple[int, int]]]):
    """
    Subject-balanced training sampler.

    Properties:
    - approximately len(recordings) samples/epoch;
    - per-subject exposure differs by at most one sample;
    - no subject appears twice in the same contrastive batch;
    - a random READY recording is chosen each time a subject is drawn;
    - emits a deterministic draw_seed with each recording index.
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
        self.by_subject = group_recording_indices_by_subject(self.recordings)
        self.subjects = sorted(self.by_subject)

        if self.batch_size < 2:
            raise ValueError("Contrastive training requires batch_size >= 2.")
        if len(self.subjects) < self.batch_size:
            raise ValueError(
                f"Only {len(self.subjects)} subjects for batch size {self.batch_size}."
            )

        target = max(self.batch_size, len(self.recordings))
        self.total_samples = (target // self.batch_size) * self.batch_size
        if self.total_samples < self.batch_size:
            self.total_samples = self.batch_size
        self.n_batches = self.total_samples // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
        rng = random.Random(_seed_from_parts(self.seed, "train_sampler", self.epoch))

        # Equal quotas; remainder is assigned to a shuffled subject subset.
        n_subjects = len(self.subjects)
        base = self.total_samples // n_subjects
        rem = self.total_samples % n_subjects

        shuffled = self.subjects.copy()
        rng.shuffle(shuffled)
        quotas = {s: base for s in self.subjects}
        for s in shuffled[:rem]:
            quotas[s] += 1

        for batch_i in range(self.n_batches):
            eligible = [s for s, q in quotas.items() if q > 0]
            if len(eligible) < self.batch_size:
                raise RuntimeError(
                    "Cannot construct unique-subject batch from remaining quotas."
                )

            # Prefer subjects with larger remaining quota; random tie breaking.
            rng.shuffle(eligible)
            eligible.sort(key=lambda s: quotas[s], reverse=True)
            chosen = eligible[: self.batch_size]

            batch = []
            for slot, s in enumerate(chosen):
                rec_idx = rng.choice(self.by_subject[s])
                draw_seed = _seed_from_parts(
                    self.seed, "sample", self.epoch, batch_i, slot, s, rec_idx
                )
                batch.append((rec_idx, draw_seed))
                quotas[s] -= 1

            yield batch


def select_deterministic_validation_recordings(
    recordings: Sequence[RecordingRef],
    subject_ids: Sequence[str],
) -> List[RecordingRef]:
    """
    Exactly one BEFORE and one AFTER READY recording per validation subject.

    Camera choice is deterministic but subject-specific, preventing a fixed
    front-camera-only validation set.
    """
    wanted = set(map(str, subject_ids))
    by_subject_state: Dict[Tuple[str, str], List[RecordingRef]] = {}

    for r in recordings:
        if r.subject_id in wanted:
            by_subject_state.setdefault((r.subject_id, r.state), []).append(r)

    selected: List[RecordingRef] = []
    for subject in sorted(wanted, key=lambda x: int(x) if x.isdigit() else x):
        for state in ("before", "after"):
            options = list(by_subject_state.get((subject, state), []))
            if not options:
                raise RuntimeError(
                    f"Stage-1 validation subject {subject} has no READY {state} recording."
                )
            options.sort(
                key=lambda r: (
                    CAMERA_ORDER.get(r.camera, 99),
                    r.fallback_ratio,
                    r.h5_path,
                )
            )
            pick = _seed_from_parts(SEED, "val_camera", subject, state) % len(options)
            selected.append(options[pick])

    return selected


class ValidationUniqueSubjectBatchSampler(Sampler[List[int]]):
    """
    Validation has two rounds: one BEFORE and one AFTER per subject.
    Within every batch each subject occurs once, avoiding same-subject negatives.
    """

    def __init__(
        self,
        recordings: Sequence[RecordingRef],
        batch_size: int,
    ):
        self.recordings = list(recordings)
        self.batch_size = int(batch_size)

        by_state: Dict[str, List[int]] = {"before": [], "after": []}
        for i, r in enumerate(self.recordings):
            if r.state not in by_state:
                raise RuntimeError(f"Unexpected validation state: {r.state}")
            by_state[r.state].append(i)

        self.batches: List[List[int]] = []
        for state in ("before", "after"):
            indices = sorted(
                by_state[state],
                key=lambda i: int(self.recordings[i].subject_id)
                if self.recordings[i].subject_id.isdigit()
                else self.recordings[i].subject_id,
            )
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) >= 2:
                    self.batches.append(batch)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        yield from self.batches
