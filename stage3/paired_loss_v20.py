from __future__ import annotations
import torch
import torch.nn.functional as F
from config_stage3 import *

def _role(state):
    x=str(state).strip().lower()
    if 'before' in x:return 0
    if 'after' in x:return 1
    return None

def paired_residual_d03_loss_v20(model,out,yn,metas):
    # FINAL predictions are used for deployable diagnostics. SBP pair training, however,
    # supervises only c against a DETACHED residual target so no SBP pair gradient can
    # alter the absolute backbone.
    pred=torch.stack([out['sbp'],out['dbp'],out['stiffness']],1)
    base_sbp=out['sbp_base_v19']
    corr={(int(ib),int(ia)):c for ib,ia,c in out.get('v19_sbp_corrections',[])}
    by_sid={}
    for i,m in enumerate(metas): by_sid.setdefault(str(m['subject_id']),[]).append(i)
    dp=[];dt=[];daux=[];sbp_corr=[];sbp_resid=[];matched=0
    for idxs in by_sid.values():
        if len(idxs)!=2: continue
        a,b=idxs;ra,rb=_role(metas[a]['state']),_role(metas[b]['state'])
        if ra is None or rb is None or ra==rb: continue
        ib,ia=(a,b) if ra==0 else (b,a)
        aux=model.predict_delta_from_pair(out,ib,ia,metas[ib],metas[ia])
        c=corr.get((ib,ia),None)
        if aux is None or c is None: continue
        target_delta=yn[ia]-yn[ib]
        base_delta=base_sbp[ia]-base_sbp[ib]
        residual=(target_delta[0]-base_delta.detach())
        dp.append(pred[ia]-pred[ib]); dt.append(target_delta); daux.append(aux)
        sbp_corr.append(c); sbp_resid.append(residual); matched+=1
    if not dp:
        z=pred.sum()*0.
        return z,{'pair_sbp_v20_residual_huber':z,'pair_dbp_v14_huber':z,'pair_stiffness_v14_huber':z,
                  'pair_abs_delta_huber':z,'pair_aux_delta_huber':z,'pair_consistency_huber':z},0,[]
    dp=torch.stack(dp);dt=torch.stack(dt);daux=torch.stack(daux)
    sbp_corr=torch.stack(sbp_corr); sbp_resid=torch.stack(sbp_resid)
    abs_t=F.smooth_l1_loss(dp,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    aux_t=F.smooth_l1_loss(daux,dt,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    con_t=F.smooth_l1_loss(dp,daux,reduction='none',beta=V14_DELTA_HUBER_BETA).mean(0)
    sbp=V20_SBP_RESIDUAL_WEIGHT*F.smooth_l1_loss(sbp_corr,sbp_resid,beta=V14_DELTA_HUBER_BETA)
    dbp=V14_ABS_DELTA_WEIGHT*abs_t[1]+V14_AUX_DELTA_WEIGHT*aux_t[1]+V14_CONSISTENCY_WEIGHT*con_t[1]
    rig=V14_ABS_DELTA_WEIGHT*abs_t[2]+V14_AUX_DELTA_WEIGHT*aux_t[2]+V14_CONSISTENCY_WEIGHT*con_t[2]
    total=V11_SBP_WEIGHT*sbp+V11_DBP_WEIGHT*dbp+V11_STIFFNESS_WEIGHT*rig
    w=dt.new_tensor([V11_SBP_WEIGHT,V11_DBP_WEIGHT,V11_STIFFNESS_WEIGHT])
    aux_rows=[(float(dt[j,0].detach().cpu()),float(dp[j,0].detach().cpu())) for j in range(len(dt))]
    return total,{'pair_sbp_v20_residual_huber':sbp,'pair_dbp_v14_huber':dbp,'pair_stiffness_v14_huber':rig,
                  'pair_abs_delta_huber':(abs_t*w).sum(),'pair_aux_delta_huber':(aux_t*w).sum(),
                  'pair_consistency_huber':(con_t*w).sum(),'pair_sbp_delta_huber':abs_t[0],
                  'pair_dbp_delta_huber':abs_t[1],'pair_stiffness_delta_huber':abs_t[2]},matched,aux_rows
