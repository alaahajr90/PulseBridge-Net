from __future__ import annotations
import torch
import torch.nn as nn
from config_stage3 import *
from model_stage3_v8_base import PulseBridgeStage3DualPathV7 as V8BaseModel, BPEncoder


class LatentSubjectStateEncoderV10(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(LATENT_DIM)
        self.temporal=nn.Sequential(nn.Conv1d(LATENT_DIM,LATENT_HIDDEN,5,padding=2),nn.GELU(),nn.Conv1d(LATENT_HIDDEN,LATENT_HIDDEN,5,padding=2),nn.GELU())
        self.time_score=nn.Linear(LATENT_HIDDEN,1)
        self.camera_emb=nn.Embedding(len(CAMERAS),CAMERA_EMBED_DIM)
        self.seg_proj=nn.Sequential(nn.Linear(LATENT_HIDDEN+CAMERA_EMBED_DIM,LATENT_HIDDEN),nn.LayerNorm(LATENT_HIDDEN),nn.GELU())
        self.seg_score=nn.Linear(LATENT_HIDDEN,1)

    def forward(self,latents,mask,camera_ids):
        B,N,T,C=latents.shape
        if (T,C)!=(LATENT_TEMPORAL_LEN,LATENT_DIM): raise RuntimeError(f'Unexpected latent shape: {tuple(latents.shape)}')
        z=self.norm(latents.reshape(B*N,T,C)).transpose(1,2)
        z=self.temporal(z).transpose(1,2)
        tw=torch.softmax(self.time_score(z).squeeze(-1),dim=1)
        seg=(z*tw.unsqueeze(-1)).sum(1).reshape(B,N,LATENT_HIDDEN)
        ce=self.camera_emb(camera_ids.clamp(min=0))
        seg=self.seg_proj(torch.cat([seg,ce],dim=-1))
        ss=self.seg_score(seg).squeeze(-1).masked_fill(~mask,-1e9)
        sw=torch.softmax(ss,dim=1)
        pooled=(seg*sw.unsqueeze(-1)).sum(1)
        return pooled,sw,seg


class AmpScaleEncoderV12(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(AMP_SUMMARY_DIM,AMP_HIDDEN),nn.LayerNorm(AMP_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT))
    def forward(self,amp_features,mask):
        m=mask.unsqueeze(-1).to(amp_features.dtype); n=m.sum(1).clamp_min(1.0)
        mean=(amp_features*m).sum(1)/n
        var=((amp_features-mean.unsqueeze(1)).square()*m).sum(1)/n
        sd=torch.sqrt(var+1e-6)
        return self.proj(torch.cat([mean,sd],dim=-1))


class PairChangePool(nn.Module):
    """Encode matched Before/After segment interactions BEFORE subject-state pooling."""
    def __init__(self,in_dim,out_dim):
        super().__init__()
        # [before, after, after-before, after*before]
        self.enc=nn.Sequential(nn.Linear(in_dim*4,V14_CHANGE_HIDDEN),nn.LayerNorm(V14_CHANGE_HIDDEN),nn.GELU(),nn.Dropout(DROPOUT),
                               nn.Linear(V14_CHANGE_HIDDEN,out_dim),nn.LayerNorm(out_dim),nn.GELU())
        self.score=nn.Sequential(nn.Linear(out_dim,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,b,a):
        if b.ndim!=2 or a.ndim!=2 or b.shape!=a.shape: raise RuntimeError('PairChangePool expects matched [M,D] tensors')
        h=self.enc(torch.cat([b,a,a-b,a*b],dim=-1))
        w=torch.softmax(self.score(h).squeeze(-1),dim=0)
        return (h*w[:,None]).sum(0),w


class PulseBridgeStage3DualPathV7(nn.Module):
    """v17 controlled experiment: v12 absolute path + pre-pooling paired change encoders.

    Absolute inference is intentionally preserved one subject-state at a time.
    During paired training, matched (camera, clip_index) segments are compared before
    subject-state pooling. BP and rigidity use distinct change pathways.
    """
    def __init__(self):
        super().__init__()
        self.base=V8BaseModel()
        self.latent_encoder=LatentSubjectStateEncoderV10()
        def make_head():
            h=nn.Sequential(nn.Linear(LATENT_HIDDEN+1,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(64,1))
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias); return h
        self.sbp_latent_residual=make_head(); self.dbp_latent_residual=make_head(); self.stiffness_latent_residual=make_head()
        self.amp_encoder=AmpScaleEncoderV12()
        def make_amp_head():
            h=nn.Sequential(nn.Linear(AMP_HIDDEN+1,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(32,1))
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias); return h
        self.sbp_amp_residual=make_amp_head(); self.dbp_amp_residual=make_amp_head(); self.stiffness_amp_residual=make_amp_head()

        # BP sees dedicated BP morphology + Stage2 segment latent + raw amplitude descriptors.
        self.bp_change_pool=PairChangePool(FEATURE_DIM+LATENT_HIDDEN+AMP_FEATURE_DIM,V14_CHANGE_HIDDEN)
        self.bp_delta_head=nn.Sequential(nn.Linear(V14_CHANGE_HIDDEN,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(64,2))

        # v18 SBP paired branch: fixed physiologically constrained morphology descriptors.
        # Descriptor extraction/standardization occurs in data_stage3 using TRAIN-only statistics.
        # No learnable waveform encoder and no attention pooling are used here, preventing the
        # representation drift observed in v17.
        self.sbp_phys_delta_head=nn.Sequential(
            nn.Linear(V18_PAIR_INPUT_DIM,V18_SBP_HIDDEN),nn.LayerNorm(V18_SBP_HIDDEN),
            nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(V18_SBP_HIDDEN,32),nn.GELU(),nn.Linear(32,1))

        # v19: small zero-initialized correction head using only stable D03=log_d1_rms.
        # Input is the exact 6D pair summary used by the diagnostic. The predicted
        # correction is applied antisymmetrically: B -= .5*c, A += .5*c.
        self.sbp_d03_correction_head=nn.Sequential(
            nn.Linear(V19_PAIR_INPUT_DIM,V19_CORR_HIDDEN),nn.LayerNorm(V19_CORR_HIDDEN),
            nn.GELU(),nn.Linear(V19_CORR_HIDDEN,1))
        nn.init.zeros_(self.sbp_d03_correction_head[-1].weight)
        nn.init.zeros_(self.sbp_d03_correction_head[-1].bias)
        # Rigidity has an independent vascular morphology change pathway.
        self.st_change_pool=PairChangePool(FEATURE_DIM+LATENT_HIDDEN,V14_CHANGE_HIDDEN)
        self.st_delta_head=nn.Sequential(nn.Linear(V14_CHANGE_HIDDEN,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(DROPOUT),nn.Linear(64,1))

    def forward(self,waves,prez_waves,mask,camera_ids,latents,amp_features,phys_features):
        out=self.base(waves,mask,camera_ids)
        z,latent_weights,latent_seg=self.latent_encoder(latents,mask,camera_ids)
        sbp0,dbp0,st0=out['sbp'],out['dbp'],out['stiffness']
        ds=self.sbp_latent_residual(torch.cat([z,sbp0[:,None]],1)).squeeze(1)
        dd=self.dbp_latent_residual(torch.cat([z,dbp0[:,None]],1)).squeeze(1)
        dr=self.stiffness_latent_residual(torch.cat([z,st0[:,None]],1)).squeeze(1)
        new=dict(out); new['sbp']=sbp0+ds; new['dbp']=dbp0+dd; new['stiffness']=st0+dr
        za=self.amp_encoder(amp_features,mask)
        new['sbp']=new['sbp']+self.sbp_amp_residual(torch.cat([za,new['sbp'][:,None]],1)).squeeze(1)
        new['dbp']=new['dbp']+self.dbp_amp_residual(torch.cat([za,new['dbp'][:,None]],1)).squeeze(1)
        new['stiffness']=new['stiffness']+self.stiffness_amp_residual(torch.cat([za,new['stiffness'][:,None]],1)).squeeze(1)
        new['bp']=torch.stack([new['sbp'],new['dbp']],1)
        new['latent_subject_state_embedding']=z; new['amp_subject_state_embedding']=za
        new['latent_segment_embedding']=latent_seg
        new['amp_segment_feature']=amp_features
        new['phys_segment_feature']=phys_features
        new['prez_waveform']=prez_waves
        new['latent_segment_weight']=latent_weights
        return new

    @staticmethod
    def _state_amp_summary(amp):
        # amp: [N,4], already scaled with TRAIN-only AmpFeatureScaler.
        # No waveform z-normalization is involved in these descriptors.
        mean=amp.mean(dim=0)
        sd=amp.std(dim=0,unbiased=False)
        return torch.cat([mean,sd],dim=0)

    @staticmethod
    def _matched_indices(meta_b,meta_a,device):
        kb={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(meta_b['cameras'],meta_b['clip_indices']))}
        ka={(str(c),int(ci)):i for i,(c,ci) in enumerate(zip(meta_a['cameras'],meta_a['clip_indices']))}
        keys=sorted(set(kb).intersection(ka))
        if len(keys)<V14_MIN_MATCHED_SEGMENTS: return None,None
        ib=torch.tensor([kb[k] for k in keys],dtype=torch.long,device=device)
        ia=torch.tensor([ka[k] for k in keys],dtype=torch.long,device=device)
        return ib,ia


    def apply_v19_coupled_sbp_correction(self,out,metas):
        """Apply pair-aware antisymmetric D03 correction to final normalized SBP.

        Requires both states of a subject in the same batch. Subjects without a valid
        exact matched pair are left unchanged. Returns diagnostic correction rows.
        """
        corrected=out['sbp'].clone()
        by_sid={}
        for i,m in enumerate(metas): by_sid.setdefault(str(m['subject_id']),[]).append(i)
        rows=[]
        for idxs in by_sid.values():
            if len(idxs)!=2: continue
            a,b=idxs
            def role(x):
                x=str(x).strip().lower()
                if 'before' in x:return 0
                if 'after' in x:return 1
                return None
            ra,rb=role(metas[a]['state']),role(metas[b]['state'])
            if ra is None or rb is None or ra==rb: continue
            ib,ia=(a,b) if ra==0 else (b,a)
            mb,ma=self._matched_indices(metas[ib],metas[ia],corrected.device)
            if mb is None: continue
            # phys_segment_feature is standardized using TRAIN-only fold statistics.
            xb=out['phys_segment_feature'][ib,mb,V19_D03_INDEX]
            xa=out['phys_segment_feature'][ia,ma,V19_D03_INDEX]
            xd=xa-xb
            v=torch.stack([xb.mean(),xb.std(unbiased=False),xa.mean(),xa.std(unbiased=False),
                           xd.mean(),xd.std(unbiased=False)])
            c=self.sbp_d03_correction_head(v).squeeze()
            corrected[ib]=corrected[ib]-0.5*c
            corrected[ia]=corrected[ia]+0.5*c
            rows.append((ib,ia,c))
        out['sbp_base_v19']=out['sbp']
        out['sbp']=corrected
        out['bp']=torch.stack([out['sbp'],out['dbp']],1)
        out['v19_sbp_corrections']=rows
        return out

    def predict_delta_from_pair(self,out,idx_b,idx_a,meta_b,meta_a):
        device=out['sbp'].device
        ib,ia=self._matched_indices(meta_b,meta_a,device)
        if ib is None: return None
        bp_b=torch.cat([out['segment_bp_feature'][idx_b,ib],out['latent_segment_embedding'][idx_b,ib],out['amp_segment_feature'][idx_b,ib]],dim=-1)
        bp_a=torch.cat([out['segment_bp_feature'][idx_a,ia],out['latent_segment_embedding'][idx_a,ia],out['amp_segment_feature'][idx_a,ia]],dim=-1)
        st_b=torch.cat([out['segment_stiffness_feature'][idx_b,ib],out['latent_segment_embedding'][idx_b,ib]],dim=-1)
        st_a=torch.cat([out['segment_stiffness_feature'][idx_a,ia],out['latent_segment_embedding'][idx_a,ia]],dim=-1)
        bpz,_=self.bp_change_pool(bp_b,bp_a); stz,_=self.st_change_pool(st_b,st_a)
        bp_delta_v14=self.bp_delta_head(bpz)
        st_delta=self.st_delta_head(stz).view(1)

        # v18 SBP: fixed standardized morphology descriptors, summarized without attention.
        # Exact camera+clip matching is retained. The representation contains state mean/SD
        # plus matched change mean/SD, so it is phase-invariant and cannot drift during training.
        ph_b=out['phys_segment_feature'][idx_b,ib]
        ph_a=out['phys_segment_feature'][idx_a,ia]
        def ms(x):
            return torch.cat([x.mean(dim=0),x.std(dim=0,unbiased=False)],dim=0)
        pair_vec=torch.cat([ms(ph_b),ms(ph_a),ms(ph_a-ph_b)],dim=0)
        sbp_phys=self.sbp_phys_delta_head(pair_vec).view(1)

        # DBP and rigidity remain exactly on their v14 auxiliary pathways.
        return torch.cat([sbp_phys,bp_delta_v14[1:2],st_delta],dim=0)
