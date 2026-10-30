from __future__ import annotations
import torch
import torch.nn.functional as F
from config_stage3 import *

def _role(state):
    x=str(state).strip().lower()
    if 'before' in x:return 0
    if 'after' in x:return 1
    return None

def paired_physmorph_loss_v18(model,out,yn,metas):
    pred=torch.stack([out['sbp'],out['dbp'],out['stiffness']],1)
    by_sid={}
    for i,m in enumerate(metas): by_sid.setdefault(str(m['subject_id']),[]).append(i)
    dp=[];dt=[];daux=[];matched=0
    for idxs in by_sid.values():
        if len(idxs)!=2: continue
        a,b=idxs;ra,rb=_role(metas[a]['state']),_role(metas[b]['state'])
        if ra is None or rb is None or ra==rb: continue
        ib,ia=(a,b) if ra==0 else (b,a)
        aux=model.predict_delta_from_pair(out,ib,ia,metas[ib],metas[ia])
        if aux is None: continue
        dp.append(pred[ia]-pred[ib]);dt.append(yn[ia]-yn[ib]);daux.append(aux);matched+=1
    if not dp:
        z=pred.sum()*0.
        return z,{'pair_sbp_v18_huber':z,'pair_dbp_v14_huber':z,'pair_stiffness_v14_huber':z,
                  'pair_abs_delta_huber':z,'pair_aux_delta_huber':z,'pair_consistency_huber':z},0,[]
    dp=torch.stack(dp);dt=torch.stack(dt);daux=torch.stack(daux)
    abs_t=F.smooth_l1_loss(dp,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    aux_t=F.smooth_l1_loss(daux,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    con_t=F.smooth_l1_loss(dp,daux,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    # v17 restores a path from paired SBP supervision to the deployable absolute outputs,
    # while the new auxiliary branch is supervised directly and consistency is modest.
    sbp=V18_SBP_ABS_DELTA_WEIGHT*abs_t[0]+V18_SBP_AUX_DELTA_WEIGHT*aux_t[0]+V18_SBP_CONSISTENCY_WEIGHT*con_t[0]
    dbp=V14_ABS_DELTA_WEIGHT*abs_t[1]+V14_AUX_DELTA_WEIGHT*aux_t[1]+V14_CONSISTENCY_WEIGHT*con_t[1]
    rig=V14_ABS_DELTA_WEIGHT*abs_t[2]+V14_AUX_DELTA_WEIGHT*aux_t[2]+V14_CONSISTENCY_WEIGHT*con_t[2]
    total=V11_SBP_WEIGHT*sbp+V11_DBP_WEIGHT*dbp+V11_STIFFNESS_WEIGHT*rig
    w=dt.new_tensor([V11_SBP_WEIGHT,V11_DBP_WEIGHT,V11_STIFFNESS_WEIGHT])
    # Return normalized SBP auxiliary pairs for validation diagnostics/checkpoint selection.
    aux_rows=[(float(dt[j,0].detach().cpu()),float(daux[j,0].detach().cpu())) for j in range(len(dt))]
    return total,{'pair_sbp_v18_huber':sbp,'pair_dbp_v14_huber':dbp,'pair_stiffness_v14_huber':rig,
                  'pair_abs_delta_huber':(abs_t*w).sum(),'pair_aux_delta_huber':(aux_t*w).sum(),
                  'pair_consistency_huber':(con_t*w).sum(),'pair_sbp_delta_huber':abs_t[0],
                  'pair_dbp_delta_huber':abs_t[1],'pair_stiffness_delta_huber':abs_t[2]},matched,aux_rows
