from __future__ import annotations
import random
from torch.utils.data import Sampler

class PairedSubjectStateBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=42):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('v11 paired training requires an even batch_size >= 2')
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        by_subject = {}
        for idx, item in enumerate(dataset.items):
            (sid, state), _ = item
            by_subject.setdefault(str(sid), []).append((idx, str(state)))
        self.pairs = []
        bad = []
        for sid, rows in sorted(by_subject.items()):
            if len(rows) != 2:
                bad.append((sid, len(rows)))
                continue
            rows = sorted(rows, key=lambda x: x[1].lower())
            self.pairs.append((rows[0][0], rows[1][0]))
        if bad:
            raise RuntimeError(f'Subjects without exactly two states: {bad[:10]}')
        if not self.pairs:
            raise RuntimeError('No subject-state pairs available')

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pairs = list(self.pairs)
        rng.shuffle(pairs)
        per_batch = self.batch_size // 2
        for i in range(0, len(pairs), per_batch):
            batch = []
            for a, b in pairs[i:i+per_batch]:
                batch.extend([a, b])
            if batch:
                yield batch

    def __len__(self):
        per_batch = self.batch_size // 2
        return (len(self.pairs) + per_batch - 1) // per_batch
