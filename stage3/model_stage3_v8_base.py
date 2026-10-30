from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_stage3 import *


def diff_pad(x, lag=1):
    d=x[:,lag:]-x[:,:-lag]
    return F.pad(d,(lag,0))


class ResBlock(nn.Module):
    def __init__(self,cin,cout,k=5,d=1,drop=0.1):
        super().__init__();p=d*(k//2)
        self.c1=nn.Conv1d(cin,cout,k,padding=p,dilation=d,bias=False);self.b1=nn.BatchNorm1d(cout)
        self.c2=nn.Conv1d(cout,cout,k,padding=p,dilation=d,bias=False);self.b2=nn.BatchNorm1d(cout)
        self.skip=nn.Identity() if cin==cout else nn.Conv1d(cin,cout,1,bias=False);self.drop=nn.Dropout(drop)
    def forward(self,x):
        y=F.gelu(self.b1(self.c1(x)));y=self.drop(y);y=self.b2(self.c2(y));return F.gelu(y+self.skip(x))


class Stream(nn.Module):
    def __init__(self,in_ch=1):
        super().__init__();self.net=nn.Sequential(
            ResBlock(in_ch,BASE_CH,7,1,DROPOUT/2),nn.MaxPool1d(2),
            ResBlock(BASE_CH,BASE_CH,5,2,DROPOUT/2),nn.MaxPool1d(2),
            ResBlock(BASE_CH,BASE_CH,3,4,DROPOUT/2))
    def forward(self,x):
        h=self.net(x)
        return torch.cat([h.mean(-1),h.amax(-1)],-1)


class BPEncoder(nn.Module):
    """BP path: raw + first derivative + multi-scale increments."""
    def __init__(self):
        super().__init__();self.raw=Stream(1);self.d1=Stream(1);self.ms=Stream(3)
        self.proj=nn.Sequential(nn.Linear(BASE_CH*2*3,FEATURE_DIM),nn.LayerNorm(FEATURE_DIM),nn.GELU(),nn.Dropout(DROPOUT))
    def forward(self,x):
        r=self.raw(x[:,None]);d1=diff_pad(x,1);a=self.d1(d1[:,None]);ms=torch.stack([diff_pad(x,1),diff_pad(x,2),diff_pad(x,4)],1);m=self.ms(ms)
        return self.proj(torch.cat([r,a,m],-1))


class StiffnessEncoder(nn.Module):
    """Stiffness path: raw + d1 + d2, kept independent from BP from the first convolution."""
    def __init__(self):
        super().__init__();self.raw=Stream(1);self.d1=Stream(1);self.d2=Stream(1)
        self.proj=nn.Sequential(nn.Linear(BASE_CH*2*3,FEATURE_DIM),nn.LayerNorm(FEATURE_DIM),nn.GELU(),nn.Dropout(DROPOUT))
    def forward(self,x):
        d1=diff_pad(x,1);d2=diff_pad(d1,1)
        return self.proj(torch.cat([self.raw(x[:,None]),self.d1(d1[:,None]),self.d2(d2[:,None])],-1))


def deterministic_quality(x,eps=1e-8):
    """Label-free deterministic quality score, one scalar per segment."""
    with torch.no_grad():
        z=x.detach()
        spec=torch.fft.rfft(z,dim=-1).abs().square()[:,1:]+eps
        p=spec/spec.sum(-1,keepdim=True)
        entropy=-(p*torch.log(p)).sum(-1)/math.log(float(p.shape[-1]))
        spectral=(1.0-entropy).clamp(0,1)
        peak=(spec.max(-1).values/spec.sum(-1)).clamp(0,1)
        ac=[]
        for lag in (8,12,16,20,24,28,32,36,40):
            if lag<z.shape[-1]: ac.append((z[:,:-lag]*z[:,lag:]).mean(-1))
        periodic=torch.stack(ac,-1).amax(-1) if ac else z.new_zeros(len(z))
        periodic=((periodic+1.0)/2.0).clamp(0,1)
        d1=diff_pad(z,1);d2=diff_pad(d1,1)
        rough=(d2.square().mean(-1)/(d1.square().mean(-1)+eps)).clamp_min(0)
        stability=torch.exp(-0.25*rough).clamp(0,1)
        q=0.35*periodic+0.30*spectral+0.20*peak+0.15*stability
        return torch.nan_to_num(q,nan=0.0,posinf=0.0,neginf=0.0).clamp(0,1)


class DeterministicHierarchicalFusion(nn.Module):
    """No learned quality gate: deterministic segment/camera weights; only feature projection is learned."""
    def __init__(self):
        super().__init__();self.camemb=nn.Embedding(len(CAMERAS),CAMERA_EMBED_DIM)
        self.proj=nn.Sequential(nn.Linear(FEATURE_DIM+CAMERA_EMBED_DIM,FUSION_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT))
    def forward(self,feat,q,camera_ids):
        z=self.proj(torch.cat([feat,self.camemb(camera_ids)],-1));segw=z.new_zeros(len(z));ctx=[];cq=[];present=[]
        for c in range(len(CAMERAS)):
            idx=(camera_ids==c).nonzero(as_tuple=False).flatten()
            if len(idx)==0:continue
            logits=((q[idx]-q[idx].mean())/QUALITY_TEMPERATURE).clamp(QUALITY_MIN_LOGIT,QUALITY_MAX_LOGIT)
            a=torch.softmax(logits,dim=0);segw[idx]=a;ctx.append((z[idx]*a[:,None]).sum(0));cq.append((q[idx]*a).sum());present.append(c)
        if not ctx:raise RuntimeError('No valid camera segments')
        C=torch.stack(ctx);cq=torch.stack(cq);cw=torch.softmax(((cq-cq.mean())/QUALITY_TEMPERATURE).clamp(QUALITY_MIN_LOGIT,QUALITY_MAX_LOGIT),dim=0)
        fused=(C*cw[:,None]).sum(0);camw=z.new_zeros(len(CAMERAS))
        for j,c in enumerate(present):camw[c]=cw[j]
        return fused,segw,camw


class BeatMorphEncoderV8(nn.Module):
    """Beat-synchronous raw+d1+d2 morphology with beat and camera-aware pooling."""
    def __init__(self,emb=64,beat_len=128,max_beats=12):
        super().__init__();self.emb=emb;self.beat_len=beat_len;self.max_beats=max_beats
        self.beat_net=nn.Sequential(nn.Conv1d(3,32,9,padding=4),nn.BatchNorm1d(32),nn.GELU(),nn.Conv1d(32,48,7,padding=3),nn.BatchNorm1d(48),nn.GELU(),nn.Conv1d(48,emb,5,padding=2),nn.BatchNorm1d(emb),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.beat_attn=nn.Sequential(nn.Linear(emb,32),nn.GELU(),nn.Linear(32,1))
        self.cam_emb=nn.Embedding(3,8)
        self.seg_attn=nn.Sequential(nn.Linear(emb+9,32),nn.GELU(),nn.Linear(32,1))
    def _zn(self,x): return (x-x.mean())/(x.std(unbiased=False)+1e-6)
    def _period(self,z):
        T=int(z.numel());lo=max(2,T//120);hi=max(lo+2,min(T//3,T//30))
        if hi<=lo+1:return max(2,T//60),z.new_tensor(0.)
        ac=torch.stack([(z[:-lag]*z[lag:]).mean() for lag in range(lo,hi+1)])
        j=int(torch.argmax(ac).item());return lo+j,torch.clamp(ac[j],-1.,1.)
    def _beats(self,x):
        import torch.nn.functional as F
        z=self._zn(x);T=int(z.numel());period,q=self._period(z)
        k=max(3,2*(period//4)+1)
        if k>=T:k=T-1 if (T-1)%2==1 else T-2
        if k<3:return None,q
        pooled=F.max_pool1d(z.view(1,1,-1),k,1,k//2).view(-1)
        cand=torch.nonzero(z>=pooled-1e-7,as_tuple=False).view(-1)
        if cand.numel()==0:return None,q
        order=cand[torch.argsort(z[cand],descending=True)]
        chosen=[];minsep=max(2,int(.55*period));half=max(4,int(.70*period))
        for idx in order.tolist():
            if idx-half<0 or idx+half>=T:continue
            if all(abs(idx-j)>=minsep for j in chosen):chosen.append(idx)
            if len(chosen)>=self.max_beats:break
        if len(chosen)<2:return None,q
        chosen.sort();out=[]
        for idx in chosen:
            b=F.interpolate(z[idx-half:idx+half+1].view(1,1,-1),size=self.beat_len,mode='linear',align_corners=False).view(-1)
            b=self._zn(b);d1=torch.zeros_like(b);d1[1:]=b[1:]-b[:-1];d2=torch.zeros_like(b);d2[1:]=d1[1:]-d1[:-1]
            out.append(torch.stack([b,d1,d2],0))
        return torch.stack(out,0),q
    def forward(self,waves,mask,cams):
        E=[];A=[]
        for j in torch.nonzero(mask,as_tuple=False).view(-1).tolist():
            beats,q=self._beats(waves[j])
            if beats is None:continue
            b=self.beat_net(beats).squeeze(-1);aw=torch.softmax(self.beat_attn(b).squeeze(-1),0);e=(aw[:,None]*b).sum(0)
            camj=cams if cams.ndim==0 else cams[j];ce=self.cam_emb(camj.long().clamp(0,2));E.append(e);A.append(torch.cat([e,ce,q.reshape(1)],0))
        if not E:return waves.new_zeros(self.emb)
        E=torch.stack(E);A=torch.stack(A);sw=torch.softmax(self.seg_attn(A).squeeze(-1),0);return (sw[:,None]*E).sum(0)

class PulseBridgeStage3DualPathV7(nn.Module):
    def __init__(self):
        super().__init__();self.bp_encoder=BPEncoder();self.st_encoder=StiffnessEncoder();self.bp_fusion=DeterministicHierarchicalFusion();self.st_fusion=DeterministicHierarchicalFusion()
        self.beat_morph=BeatMorphEncoderV8(emb=64)
        self.bp_morph_fuse=nn.Sequential(nn.Linear(FUSION_HIDDEN+64,FUSION_HIDDEN),nn.LayerNorm(FUSION_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT))
        self.sbp_head=nn.Sequential(nn.Linear(FUSION_HIDDEN,HEAD_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(HEAD_HIDDEN,1))
        self.dbp_head=nn.Sequential(nn.Linear(FUSION_HIDDEN,HEAD_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(HEAD_HIDDEN,1))
        self.st_head=nn.Sequential(nn.Linear(FUSION_HIDDEN,HEAD_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(HEAD_HIDDEN,1))
        self.seg_bp=nn.Linear(FEATURE_DIM,2);self.seg_st=nn.Linear(FEATURE_DIM,1)
    def forward(self,waves,mask,camera_ids):
        B,N,T=waves.shape;flat=waves.reshape(B*N,T);valid=mask.reshape(B*N);camflat=camera_ids.reshape(B*N);x=flat[valid];cams=camflat[valid]
        q=deterministic_quality(x);bpfeat=self.bp_encoder(x);stfeat=self.st_encoder(x)
        bpseg=self.seg_bp(bpfeat);stseg=self.seg_st(stfeat)[:,0];bpsegfull=waves.new_zeros(B*N,2);stsegfull=waves.new_zeros(B*N);qfull=waves.new_zeros(B*N);bpw=waves.new_zeros(B*N);stw=waves.new_zeros(B*N)
        bpsegfull[valid]=bpseg;stsegfull[valid]=stseg;qfull[valid]=q
        bpout=[];stout=[];bpcw=[];stcw=[];cursor=0
        for i,n in enumerate(mask.sum(1).tolist()):
            n=int(n);bpf,bpsw,bpc=self.bp_fusion(bpfeat[cursor:cursor+n],q[cursor:cursor+n],cams[cursor:cursor+n]);stf,stsw,stc=self.st_fusion(stfeat[cursor:cursor+n],q[cursor:cursor+n],cams[cursor:cursor+n])
            morph=self.beat_morph(waves[i],mask[i],cams[i])
            bpf2=self.bp_morph_fuse(torch.cat([bpf,morph],dim=0))
            sbp=self.sbp_head(bpf2)[0];dbp=self.dbp_head(bpf2)[0]
            bpout.append(torch.stack([sbp,dbp]));stout.append(self.st_head(stf)[0]);pos=mask[i].nonzero(as_tuple=False).flatten()+i*N;bpw[pos]=bpsw;stw[pos]=stsw;bpcw.append(bpc);stcw.append(stc);cursor+=n
        bp=torch.stack(bpout);st=torch.stack(stout)
        # v14 diagnostic/change path needs the dedicated morphology features BEFORE
        # hierarchical subject-state pooling.  These tensors do not alter the absolute path.
        bpfeatfull=waves.new_zeros(B*N,FEATURE_DIM); stfeatfull=waves.new_zeros(B*N,FEATURE_DIM)
        bpfeatfull[valid]=bpfeat; stfeatfull[valid]=stfeat
        return {'bp':bp,'sbp':bp[:,0],'dbp':bp[:,1],'stiffness':st,
                'segment_bp':bpsegfull.view(B,N,2),'segment_stiffness':stsegfull.view(B,N),'segment_valid':mask,
                'segment_bp_feature':bpfeatfull.view(B,N,FEATURE_DIM),
                'segment_stiffness_feature':stfeatfull.view(B,N,FEATURE_DIM),
                'segment_quality_scalar':qfull.view(B,N),'segment_weight_bp':bpw.view(B,N),'segment_weight_stiffness':stw.view(B,N),
                'camera_weight_bp':torch.stack(bpcw),'camera_weight_stiffness':torch.stack(stcw)}
