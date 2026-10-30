from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import csv, h5py, numpy as np, torch
from torch.utils.data import Dataset
from config_stage3 import *
from io_utils import sha256_file

@dataclass(frozen=True)
class RecordingRef:
    h5_path: str
    subject_id: str
    state: str
    camera: str
    n_clips: int

def load_stage3_manifest():
    if not STAGE3_QC_MANIFEST.exists():
        raise FileNotFoundError(STAGE3_QC_MANIFEST)
    got=sha256_file(STAGE3_QC_MANIFEST)
    if got!=EXPECTED_STAGE3_QC_MANIFEST_SHA256:
        raise RuntimeError(f"Stage-3 QC manifest SHA mismatch: {got} != {EXPECTED_STAGE3_QC_MANIFEST_SHA256}")
    rows=list(csv.DictReader(open(STAGE3_QC_MANIFEST,newline="",encoding="utf-8")))
    if len(rows)!=EXPECTED_STAGE3_TOTAL_RECORDINGS: raise RuntimeError(f"Stage-3 manifest row count {len(rows)} != 3600")
    ready=[r for r in rows if str(r.get("status","")).upper()=="READY"]
    hold=[r for r in rows if str(r.get("status","")).upper()=="HOLD"]
    if len(ready)!=EXPECTED_STAGE3_READY_RECORDINGS or len(hold)!=EXPECTED_STAGE3_HOLD_RECORDINGS:
        raise RuntimeError(f"Stage-3 READY/HOLD mismatch: {len(ready)}/{len(hold)}")
    refs=[]
    for r in ready:
        hp=Path(r["h5_path"])
        if not hp.exists(): raise FileNotFoundError(f"READY H5 missing: {hp}")
        n_clips=int(float(r.get("n_clips",0))) if r.get("n_clips","")!="" else -1
        refs.append(RecordingRef(str(hp),str(r["subject_id"]),str(r["state"]),str(r["camera"]),n_clips))
    if len({r.subject_id for r in refs})!=EXPECTED_STAGE3_READY_SUBJECTS: raise RuntimeError("Stage-3 READY subject count mismatch")
    if len({(r.subject_id,r.state) for r in refs})!=EXPECTED_STAGE3_READY_SUBJECT_STATES: raise RuntimeError("Stage-3 READY subject-state count mismatch")
    return refs,hold,got

def recordings_for_subjects(refs,subjects):
    keep=set(map(str,subjects));return [r for r in refs if r.subject_id in keep]

def selected_clip_indices(n_clips,n=SEGMENTS_PER_RECORDING):
    if n_clips<=0: raise RuntimeError(f"Invalid n_clips={n_clips}")
    if n_clips<=n:return np.arange(n_clips,dtype=np.int64)
    return np.unique(np.rint(np.linspace(0,n_clips-1,n)).astype(np.int64))

def read_labels_for_allowed_refs(refs,allowed_subjects):
    allowed=set(map(str,allowed_subjects));out={}
    for r in refs:
        if r.subject_id not in allowed: continue
        key=(r.subject_id,r.state)
        with h5py.File(r.h5_path,"r") as h:
            c=h["labels"]["canonical"].attrs
            vals=np.asarray([float(c[SBP_LABEL]),float(c[DBP_LABEL]),float(c[STIFFNESS_LABEL])],dtype=np.float64)
        if not np.isfinite(vals).all(): raise RuntimeError(f"Non-finite Stage-3 labels: {key}")
        if key in out and not np.allclose(out[key],vals,rtol=0,atol=1e-6): raise RuntimeError(f"Label disagreement across cameras for {key}")
        out[key]=vals
    return out

class TargetScaler:
    names=("sbp","dbp","stiffness")
    def fit(self,labels_by_subject_state,train_subjects):
        train=set(map(str,train_subjects));arr=[v for (sid,_),v in labels_by_subject_state.items() if sid in train]
        if not arr: raise RuntimeError("No train labels for scaler.")
        a=np.stack(arr).astype(np.float64);self.mean=a.mean(0);self.std=a.std(0);self.std[self.std<1e-8]=1.0;return self
    def transform_np(self,y):return (np.asarray(y)-self.mean)/self.std
    def inverse_np(self,y):return np.asarray(y)*self.std+self.mean
    def state_dict(self):return {"names":list(self.names),"mean":self.mean.tolist(),"std":self.std.tolist()}
    @classmethod
    def from_state(cls,d):o=cls();o.mean=np.asarray(d["mean"],dtype=float);o.std=np.asarray(d["std"],dtype=float);return o


def raw_amp_features_np(x):
    """Four scale descriptors computed BEFORE per-segment z-normalization."""
    x=np.asarray(x,dtype=np.float64)
    d1=np.diff(x)
    std=float(np.std(x))
    rms=float(np.sqrt(np.mean(x*x)))
    ptp=float(np.max(x)-np.min(x))
    d1rms=float(np.sqrt(np.mean(d1*d1))) if d1.size else 0.0
    vals=np.asarray([std,rms,ptp,d1rms],dtype=np.float64)
    vals=np.log(np.maximum(vals,AMP_EPS))
    if not np.isfinite(vals).all(): raise RuntimeError("Non-finite raw amplitude features")
    return vals


class AmpFeatureScaler:
    """TRAIN-only scaler for pre-normalization rPPG scale descriptors."""
    def fit(self,cache_dir,refs,train_subjects):
        train=set(map(str,train_subjects)); valid={(r.subject_id,r.state,r.camera) for r in refs if r.subject_id in train}
        arr=[]; cache_dir=Path(cache_dir)
        for p in sorted(cache_dir.glob("*.npz")):
            with np.load(p,allow_pickle=False) as z:
                sid=str(z["subject_id"].item()); state=str(z["state"].item()); camera=str(z["camera"].item())
                if (sid,state,camera) not in valid: continue
                for x in z["rppg"]: arr.append(raw_amp_features_np(x))
        if not arr: raise RuntimeError("No TRAIN rPPG segments for AmpFeatureScaler")
        a=np.stack(arr); self.mean=a.mean(0); self.std=a.std(0); self.std[self.std<1e-8]=1.0
        return self
    def transform_np(self,x): return ((np.asarray(x)-self.mean)/self.std).astype(np.float32)
    def state_dict(self): return {"names":list(AMP_FEATURE_NAMES),"mean":self.mean.tolist(),"std":self.std.tolist()}
    @classmethod
    def from_state(cls,d):
        o=cls(); o.mean=np.asarray(d["mean"],dtype=float); o.std=np.asarray(d["std"],dtype=float); return o



def raw_phys_morph_features_np(x):
    """Fixed, phase-invariant PRE-Z pulse descriptors. No labels and no learned weights.

    The first four retain scale; remaining descriptors summarize distribution,
    slope/curvature and periodicity without requiring pointwise Before/After alignment.
    """
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    xc=x-float(np.mean(x)); eps=AMP_EPS
    sd=float(np.std(xc)); rms=float(np.sqrt(np.mean(xc*xc))); ptp=float(np.ptp(xc))
    d1=np.diff(xc); d2=np.diff(xc,n=2)
    d1r=float(np.sqrt(np.mean(d1*d1))) if d1.size else 0.0
    z=xc/(sd+eps)
    skew=float(np.mean(z**3)); kurt=float(np.mean(z**4)-3.0)
    d1abs=float(np.mean(np.abs(d1))/(sd+eps)) if d1.size else 0.0
    d2abs=float(np.mean(np.abs(d2))/(sd+eps)) if d2.size else 0.0
    q10,q25,q75,q90=np.quantile(z,[.10,.25,.75,.90])
    iqr=float(q75-q25)
    asym=float(((q90-q75)-(q25-q10))/(abs(q90-q10)+eps))
    # Normalized autocorrelation; ignore lag 0. Search a broad pulse-period range
    # in samples, intentionally avoiding camera-FPS assumptions.
    ac=np.correlate(z,z,mode='full')[len(z)-1:]
    ac=ac/(ac[0]+eps)
    lo=max(2,int(round(len(z)*0.08))); hi=max(lo+1,int(round(len(z)*0.80)))
    hi=min(hi,len(ac))
    if hi>lo:
        j=lo+int(np.argmax(ac[lo:hi])); acpk=float(ac[j]); lag=float(j/max(len(z)-1,1))
    else:
        acpk=0.0; lag=0.0
    vals=np.asarray([np.log(max(sd,eps)),np.log(max(rms,eps)),np.log(max(ptp,eps)),np.log(max(d1r,eps)),
                     skew,kurt,d1abs,d2abs,q10,q25,q75,q90,iqr,asym,acpk,lag],dtype=np.float64)
    if not np.isfinite(vals).all(): raise RuntimeError("Non-finite v18 physiology morphology features")
    return vals


class PhysMorphFeatureScaler:
    """TRAIN-only standardization for the fixed v18 morphology descriptor vector."""
    def fit(self,cache_dir,refs,train_subjects):
        train=set(map(str,train_subjects)); valid={(r.subject_id,r.state,r.camera) for r in refs if r.subject_id in train}
        arr=[]; cache_dir=Path(cache_dir)
        for p in sorted(cache_dir.glob("*.npz")):
            with np.load(p,allow_pickle=False) as z:
                sid=str(z["subject_id"].item()); state=str(z["state"].item()); camera=str(z["camera"].item())
                if (sid,state,camera) not in valid: continue
                for x in z["rppg"]: arr.append(raw_phys_morph_features_np(x))
        if not arr: raise RuntimeError("No TRAIN rPPG segments for PhysMorphFeatureScaler")
        a=np.stack(arr); self.mean=a.mean(0); self.std=a.std(0); self.std[self.std<1e-8]=1.0
        return self
    def transform_np(self,x): return ((np.asarray(x)-self.mean)/self.std).astype(np.float32)
    def state_dict(self): return {"names":list(V18_PHYS_FEATURE_NAMES),"mean":self.mean.tolist(),"std":self.std.tolist()}


class Stage3SubjectStateDataset(Dataset):
    """One item = one (subject,state). All available cached camera/segment rPPGs stay grouped."""
    def __init__(self,cache_dir,refs,labels_by_subject_state,scaler,subjects,amp_scaler=None,phys_scaler=None):
        subjects=set(map(str,subjects));valid={(r.subject_id,r.state,r.camera):r for r in refs if r.subject_id in subjects}
        groups=defaultdict(list);cache_dir=Path(cache_dir);self.latent_cache_dir=LATENT_CACHE_ROOT/cache_dir.name
        if not (cache_dir/"CACHE_COMPLETE.json").exists(): raise RuntimeError(f"Required development cache incomplete: {cache_dir}")
        if not (self.latent_cache_dir/"CACHE_COMPLETE.json").exists(): raise RuntimeError(f"Required v10 latent cache incomplete: {self.latent_cache_dir}")
        for p in sorted(cache_dir.glob("*.npz")):
            with np.load(p,allow_pickle=False) as z:
                sid=str(z["subject_id"].item());state=str(z["state"].item());camera=str(z["camera"].item())
                if (sid,state,camera) not in valid: continue
                waves=z["rppg"];clip_indices=z["clip_indices"]
                for i in range(len(waves)):groups[(sid,state)].append((str(p),int(i),camera,int(clip_indices[i])))
        self.items=[]
        for key in sorted(groups):
            if key not in labels_by_subject_state: raise RuntimeError(f"Missing label for cached subject-state {key}")
            self.items.append((key,groups[key]))
        self.labels=labels_by_subject_state;self.scaler=scaler;self.amp_scaler=amp_scaler;self.phys_scaler=phys_scaler
    def __len__(self):return len(self.items)
    def __getitem__(self,idx):
        (sid,state),segs=self.items[idx];waves=[];prez_waves=[];latents=[];amp_features=[];phys_features=[];cameras=[];camera_ids=[];clip_indices=[]
        for path,segi,camera,ci in segs:
            with np.load(path,allow_pickle=False) as z:x=z["rppg"][segi].astype(np.float32)
            amp_raw=raw_amp_features_np(x)
            phys_raw=raw_phys_morph_features_np(x)
            # v17 scale-aware waveform: remove only DC offset, then divide by a TRAIN-only
            # fold reference scale (geometric mean segment std). No per-segment scale normalization.
            ref_scale=float(np.exp(self.amp_scaler.mean[0])) if self.amp_scaler is not None else 1.0
            if not np.isfinite(ref_scale) or ref_scale<=AMP_EPS: raise RuntimeError("Invalid TRAIN pre-z reference scale")
            prez=((x.astype(np.float64)-float(np.mean(x)))/ref_scale).astype(np.float32)
            if not np.isfinite(prez).all(): raise RuntimeError("Non-finite v17 pre-z waveform")
            amp=self.amp_scaler.transform_np(amp_raw) if self.amp_scaler is not None else amp_raw.astype(np.float32)
            phys=self.phys_scaler.transform_np(phys_raw) if self.phys_scaler is not None else phys_raw.astype(np.float32)
            lp=self.latent_cache_dir/Path(path).name
            if not lp.exists(): raise RuntimeError(f"Missing v10 latent sidecar: {lp}")
            with np.load(lp,allow_pickle=False) as lz:
                if int(lz["clip_indices"][segi]) != int(ci): raise RuntimeError(f"Latent/rPPG clip-index misalignment: {lp}")
                lat=lz["f7"][segi].astype(np.float32)
            if lat.shape!=(LATENT_TEMPORAL_LEN,LATENT_DIM): raise RuntimeError(f"Bad v10 latent shape {lat.shape} in {lp}")
            x=(x-x.mean())/(x.std()+1e-8)
            if camera not in CAMERA_TO_INDEX: raise RuntimeError(f"Unknown camera: {camera}")
            waves.append(x);prez_waves.append(prez);latents.append(lat);amp_features.append(amp);phys_features.append(phys);cameras.append(camera);camera_ids.append(CAMERA_TO_INDEX[camera]);clip_indices.append(ci)
        raw=self.labels[(sid,state)].astype(np.float32);yn=self.scaler.transform_np(raw).astype(np.float32)
        return {"waves":torch.from_numpy(np.stack(waves)),"prez_waves":torch.from_numpy(np.stack(prez_waves)),"latents":torch.from_numpy(np.stack(latents)),"amp_features":torch.from_numpy(np.stack(amp_features)),"phys_features":torch.from_numpy(np.stack(phys_features)),"camera_ids":torch.tensor(camera_ids,dtype=torch.long),
                "target_norm":torch.from_numpy(yn),"target_raw":torch.from_numpy(raw),"subject_id":sid,"state":state,
                "cameras":cameras,"clip_indices":clip_indices}

def subject_state_collate(batch):
    B=len(batch);max_n=max(x["waves"].shape[0] for x in batch);T=batch[0]["waves"].shape[1]
    waves=torch.zeros(B,max_n,T,dtype=torch.float32);prez_waves=torch.zeros(B,max_n,T,dtype=torch.float32);latents=torch.zeros(B,max_n,LATENT_TEMPORAL_LEN,LATENT_DIM,dtype=torch.float32);amp_features=torch.zeros(B,max_n,AMP_FEATURE_DIM,dtype=torch.float32);phys_features=torch.zeros(B,max_n,V18_PHYS_FEATURE_DIM,dtype=torch.float32);mask=torch.zeros(B,max_n,dtype=torch.bool)
    camera_ids=torch.full((B,max_n),-1,dtype=torch.long);metas=[];yn=[];yr=[]
    for i,x in enumerate(batch):
        n=x["waves"].shape[0];waves[i,:n]=x["waves"];prez_waves[i,:n]=x["prez_waves"];latents[i,:n]=x["latents"];amp_features[i,:n]=x["amp_features"];phys_features[i,:n]=x["phys_features"];mask[i,:n]=True;camera_ids[i,:n]=x["camera_ids"]
        yn.append(x["target_norm"]);yr.append(x["target_raw"])
        metas.append({"subject_id":x["subject_id"],"state":x["state"],"cameras":x["cameras"],"clip_indices":x["clip_indices"],"n_segments":n})
    return waves,prez_waves,latents,amp_features,phys_features,mask,camera_ids,torch.stack(yn),torch.stack(yr),metas
