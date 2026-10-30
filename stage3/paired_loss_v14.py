from __future__ import annotations
import torch
import torch.nn.functional as F
from config_stage3 import *

def _role(state):
    x=str(state).strip().lower()
    if 'before' in x:return 0
    if 'after' in x:return 1
    return None

def _weighted_mean(v):
    w=v.new_tensor([V11_SBP_WEIGHT,V11_DBP_WEIGHT,V11_STIFFNESS_WEIGHT]); return (v*w).sum()

def paired_prepool_loss_v14(model,out,yn,metas):
    pred=torch.stack([out['sbp'],out['dbp'],out['stiffness']],1)
    by_sid={}
    for i,m in enumerate(metas):by_sid.setdefault(str(m['subject_id']),[]).append(i)
    dp=[];dt=[];daux=[];matched_pairs=0
    for idxs in by_sid.values():
        if len(idxs)!=2:continue
        a,b=idxs;ra,rb=_role(metas[a]['state']),_role(metas[b]['state'])
        if ra is None or rb is None or ra==rb:continue
        ib,ia=(a,b) if ra==0 else (b,a)
        aux=model.predict_delta_from_pair(out,ib,ia,metas[ib],metas[ia])
        if aux is None:continue
        dp.append(pred[ia]-pred[ib]);dt.append(yn[ia]-yn[ib]);daux.append(aux);matched_pairs+=1
    if not dp:
        zero=pred.sum()*0.0
        return zero,{'pair_abs_delta_huber':zero,'pair_aux_delta_huber':zero,'pair_consistency_huber':zero,
                     'pair_sbp_delta_huber':zero,'pair_dbp_delta_huber':zero,'pair_stiffness_delta_huber':zero},0
    dp=torch.stack(dp);dt=torch.stack(dt);daux=torch.stack(daux)
    abs_t=F.smooth_l1_loss(dp,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    aux_t=F.smooth_l1_loss(daux,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    con_t=F.smooth_l1_loss(dp,daux,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    abs_loss=_weighted_mean(abs_t);aux_loss=_weighted_mean(aux_t);con_loss=_weighted_mean(con_t)
    total=V14_ABS_DELTA_WEIGHT*abs_loss+V14_AUX_DELTA_WEIGHT*aux_loss+V14_CONSISTENCY_WEIGHT*con_loss
    return total,{'pair_abs_delta_huber':abs_loss,'pair_aux_delta_huber':aux_loss,'pair_consistency_huber':con_loss,
                  'pair_sbp_delta_huber':abs_t[0],'pair_dbp_delta_huber':abs_t[1],'pair_stiffness_delta_huber':abs_t[2]},matched_pairs
