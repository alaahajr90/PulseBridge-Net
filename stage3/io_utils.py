from __future__ import annotations
from pathlib import Path
import csv, hashlib, json, os, tempfile, torch

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def write_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

def atomic_torch_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def torch_load_full(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def read_csv_rows(path):
    path=Path(path)
    if not path.exists(): return []
    rows=[]
    with open(path,newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q={}
            for k,v in r.items():
                try: q[k]=float(v)
                except Exception: q[k]=v
            if "epoch" in q:
                try:q["epoch"]=int(float(q["epoch"]))
                except Exception:pass
            if "fold" in q:
                try:q["fold"]=int(float(q["fold"]))
                except Exception:pass
            rows.append(q)
    return rows
