from __future__ import annotations
from paired_sampler_v11 import PairedSubjectStateBatchSampler
from paired_loss_v20 import paired_residual_d03_loss_v20
import argparse, logging, random, statistics, sys, gc
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from config_stage3 import *
from protocol_stage3 import load_frozen_protocol
from data_stage3 import *
from model_stage3 import PulseBridgeStage3DualPathV7
from losses_stage3 import DualPathLossV11
from metrics_stage3 import all_target_metrics, subject_cluster_bootstrap, rows_from_batch, aggregate_segment_rows_to_recordings
from runtime_stage3 import resolve_runtime
from training_contract import build_contract
from io_utils import *


def parse_args():
    p = argparse.ArgumentParser('PulseBridge Stage3 v20 Frozen-Residual-D03 — DEVELOPMENT ONLY')
    p.add_argument('--fold', type=int, choices=range(1,11), default=None)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    try: torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception: pass


def logger_setup():
    d = OUTPUT_ROOT/'logs'; d.mkdir(parents=True, exist_ok=True)
    p = d/f"stage3v20_frozenresiduald03_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    l = logging.getLogger('stage3v20_frozenresiduald03'); l.setLevel(logging.INFO); l.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(p, encoding='utf-8')):
        h.setFormatter(fmt); l.addHandler(h)
    return l, p


def make_loader(ds, batch, training):
    kw = {'batch_size':batch,'shuffle':training,'num_workers':NUM_WORKERS,
          'pin_memory':PIN_MEMORY and torch.cuda.is_available(),'collate_fn':subject_state_collate,
          'persistent_workers':PERSISTENT_WORKERS and NUM_WORKERS>0}
    if NUM_WORKERS>0: kw['prefetch_factor']=PREFETCH_FACTOR
    # v19 correction is pair-aware, so both TRAIN and VAL loaders preserve complete
    # Before/After subject pairs in each batch. No subjects are dropped.
    kw.pop('batch_size',None); kw.pop('shuffle',None)
    kw['batch_sampler'] = PairedSubjectStateBatchSampler(ds, batch, seed=SEED)
    return DataLoader(ds, **kw)


def run_epoch(model, dl, crit, opt, device, scaler, training, collect_details=False):
    model.train(training)
    loss_sum=0.; n=0; rows=[]; segment_rows=[]; parts_sum={}; aux_sbp_pairs=[]
    ctx=torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for waves,prez_waves,latents,amp_features,phys_features,mask,camera_ids,yn,yraw,meta in dl:
            waves=waves.to(device,non_blocking=True); prez_waves=prez_waves.to(device,non_blocking=True); latents=latents.to(device,non_blocking=True); amp_features=amp_features.to(device,non_blocking=True); phys_features=phys_features.to(device,non_blocking=True)
            mask=mask.to(device); camera_ids=camera_ids.to(device); yn=yn.to(device)
            if training: opt.zero_grad(set_to_none=True)

            out=model(waves,prez_waves,mask,camera_ids,latents,amp_features,phys_features)
            out=model.apply_v19_coupled_sbp_correction(out,meta)

            # v20 isolation: absolute SBP loss sees the UNCORRECTED base prediction.
            # Therefore correction-head gradients cannot enter through absolute loss.
            out_abs=dict(out)
            out_abs['sbp']=out['sbp_base_v19']
            out_abs['bp']=torch.stack([out_abs['sbp'],out_abs['dbp']],1)
            loss,parts=crit(out_abs,yn)

            # v20 SBP pair target = true delta - DETACHED base delta. The only SBP
            # pair gradient is into the D03 correction head; backbone is isolated.
            pair_loss,pair_parts,npairs,aux_pairs=paired_residual_d03_loss_v20(model,out,yn,meta)
            aux_sbp_pairs.extend(aux_pairs)
            if training:
                loss=loss+V14_PAIR_WEIGHT*pair_loss

            parts=dict(parts)
            parts.update(pair_parts)
            parts['paired_total_loss']=pair_loss

            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite Stage3-v20 loss')
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP_NORM); opt.step()

            predn=torch.stack([out['sbp'],out['dbp'],out['stiffness']],1).detach().cpu().numpy()
            pred=scaler.inverse_np(predn); truth=yraw.numpy()
            cbp=out['camera_weight_bp'].detach().cpu().numpy(); cst=out['camera_weight_stiffness'].detach().cpu().numpy()
            rows.extend(rows_from_batch(meta,truth,pred,cbp,cst))

            if collect_details:
                bpseg=out['segment_bp'].detach().cpu().numpy(); stseg=out['segment_stiffness'].detach().cpu().numpy()
                msk=out['segment_valid'].detach().cpu().numpy(); wbp=out['segment_weight_bp'].detach().cpu().numpy()
                wst=out['segment_weight_stiffness'].detach().cpu().numpy(); q=out['segment_quality_scalar'].detach().cpu().numpy()
                for i,m in enumerate(meta):
                    vi=np.flatnonzero(msk[i])
                    segn=np.column_stack([bpseg[i,vi,0],bpseg[i,vi,1],stseg[i,vi]])
                    segphys=scaler.inverse_np(segn)
                    for j,phys in enumerate(segphys):
                        segment_rows.append({'subject_id':m['subject_id'],'state':m['state'],'camera':m['cameras'][j],
                        'clip_index':int(m['clip_indices'][j]),'true_sbp':float(truth[i,0]),'true_dbp':float(truth[i,1]),
                        'true_stiffness':float(truth[i,2]),'pred_sbp':float(phys[0]),'pred_dbp':float(phys[1]),
                        'pred_stiffness':float(phys[2]),'fusion_segment_weight_bp':float(wbp[i,vi[j]]),
                        'fusion_segment_weight_stiffness':float(wst[i,vi[j]]),'quality_score':float(q[i,vi[j]])})

            b=len(meta); loss_sum+=float(loss.item())*b; n+=b
            for k,v in parts.items():
                parts_sum[k]=parts_sum.get(k,0.)+float(v.item() if torch.is_tensor(v) else v)*b

    metrics=all_target_metrics(rows)
    metrics.update({f'normalized_{k}_loss':v/max(n,1) for k,v in parts_sum.items()})
    if aux_sbp_pairs:
        yt=np.asarray([x[0] for x in aux_sbp_pairs],float)*float(scaler.std[0])
        yp=np.asarray([x[1] for x in aux_sbp_pairs],float)*float(scaler.std[0])
        metrics['sbp_aux_delta_n']=int(len(yt))
        metrics['sbp_aux_delta_mae']=float(np.mean(np.abs(yp-yt)))
        metrics['sbp_aux_delta_r']=float(np.corrcoef(yt,yp)[0,1]) if len(yt)>2 and np.std(yt)>1e-12 and np.std(yp)>1e-12 else float('nan')
        vx=np.var(yt);vy=np.var(yp);metrics['sbp_aux_delta_ccc']=float(2*np.cov(yt,yp,ddof=0)[0,1]/(vx+vy+(yt.mean()-yp.mean())**2+1e-12))
        metrics['sbp_aux_delta_std_ratio']=float(np.std(yp)/(np.std(yt)+1e-12))
        metrics['sbp_aux_delta_bias']=float(np.mean(yp-yt))
    else:
        for q in ('n','mae','r','ccc','std_ratio','bias'): metrics['sbp_aux_delta_'+q]=0 if q=='n' else float('nan')
    rec=aggregate_segment_rows_to_recordings(segment_rows) if collect_details else []
    return loss_sum/max(n,1),metrics,rows,segment_rows,rec


def bp_error_score(row):
    return .40*(row['val_sbp_mae']/15.)+.25*(row['val_sbp_rmse']/20.)+.20*(row['val_dbp_mae']/8.)+.15*(row['val_dbp_rmse']/10.)


def delta_error_score(row, scaler):
    # Dimensionless paired error using fold-specific TRAIN scaler only.
    s=np.asarray(scaler.std,dtype=float)
    return (V11_SBP_WEIGHT*(row['val_sbp_delta_mae']/s[0])+
            V11_DBP_WEIGHT*(row['val_dbp_delta_mae']/s[1])+
            V11_STIFFNESS_WEIGHT*(row['val_stiffness_delta_mae']/s[2]))


def selection_score(row, scaler):
    # v20 selects on absolute BP plus the FINAL corrected SBP delta (not an auxiliary head).
    row['val_bp_error_score']=bp_error_score(row)
    row['val_final_sbp_delta_error_score']=row['val_sbp_delta_mae']/float(scaler.std[0])
    row['val_selection_score']=(V20_SELECTION_ABS_WEIGHT*row['val_bp_error_score']+
                                V20_SELECTION_FINAL_SBP_DELTA_WEIGHT*row['val_final_sbp_delta_error_score'])
    row['val_delta_error_score']=delta_error_score(row,scaler)
    return row['val_selection_score']

def _safe(v, fallback=-1e9):
    try:
        return float(v) if np.isfinite(float(v)) else fallback
    except Exception:
        return fallback


def better(row,best,scaler):
    rs=selection_score(row,scaler)
    if best is None: return True
    bs=best.get('val_selection_score')
    if bs is None: bs=selection_score(best,scaler)
    if rs<bs-MIN_DELTA: return True
    if abs(rs-bs)<=MIN_DELTA:
        c=.60*_safe(row['val_sbp_ccc'])+.40*_safe(row['val_dbp_ccc'])
        bc=.60*_safe(best['val_sbp_ccc'])+.40*_safe(best['val_dbp_ccc'])
        if c>bc+1e-6: return True
        if abs(c-bc)<=1e-6:
            dr=_safe(row.get('val_sbp_delta_r'))
            bdr=_safe(best.get('val_sbp_delta_r'))
            return dr>bdr+1e-6
    return False


def save_ckpt(path,model,opt,sch,epoch,row,scaler,fold,contract,best,stale):
    atomic_torch_save({'stage':V11_VERSION,'fold':fold,'epoch':epoch,'model':model.state_dict(),
        'optimizer':opt.state_dict() if opt else None,'scheduler':sch.state_dict() if sch else None,
        'metrics':row,'target_scaler':scaler.state_dict(),'contract':contract,'best':best,'stale':stale,
        'input':'v18 absolute backbone + v20 isolated D03 residual correction (base detached from SBP pair loss); DBP/RIG v14 paired paths',
        'outputs':['SBP','DBP','dataset-native rigidity'],'blind_used':False},path)


def one_fold(fi,refs,labels,batch,device,log,contract,resume):
    k=int(fi['fold']); d=OUTPUT_ROOT/'folds'/f'fold_{k:02d}'; ck=d/'checkpoints'
    d.mkdir(parents=True,exist_ok=True); ck.mkdir(parents=True,exist_ok=True); done=d/'COMPLETED.json'
    if done.exists():
        z=load_json(done)
        if not resume or z['contract']!=contract: raise RuntimeError(f'Fold {k} existing v20 result incompatible')
        log.info('Fold %02d already complete; reusing.',k); return z['summary']

    sc=TargetScaler().fit(labels,fi['train_subjects']); save_json(d/'target_scaler.json',sc.state_dict())
    cache=CACHE_ROOT/f'fold_{k:02d}'
    amp_sc=AmpFeatureScaler().fit(cache,refs,fi['train_subjects']); save_json(d/'amp_feature_scaler.json',amp_sc.state_dict())
    phys_sc=PhysMorphFeatureScaler().fit(cache,refs,fi['train_subjects']); save_json(d/'phys_morph_feature_scaler.json',phys_sc.state_dict())
    trds=Stage3SubjectStateDataset(cache,refs,labels,sc,fi['train_subjects'],amp_sc,phys_sc)
    vads=Stage3SubjectStateDataset(cache,refs,labels,sc,fi['val_subjects'],amp_sc,phys_sc)
    tr=make_loader(trds,batch,True); va=make_loader(vads,batch,False)
    model=PulseBridgeStage3DualPathV7().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=LR_SCHEDULER_FACTOR,
                                                   patience=LR_SCHEDULER_PATIENCE,min_lr=LR_MIN)
    crit=DualPathLossV11(); hist=[]; best=None; stale=0; start=1

    if resume and (ck/'last.pt').exists():
        z=torch_load_full(ck/'last.pt')
        if z['contract']!=contract: raise RuntimeError('Unsafe v18 resume: contract mismatch')
        model.load_state_dict(z['model']); opt.load_state_dict(z['optimizer']); sch.load_state_dict(z['scheduler'])
        start=z['epoch']+1; best=z['best']; stale=z['stale']; hist=read_csv_rows(d/'training_history.csv')
        log.info('Resuming fold %02d at epoch %d',k,start)

    log.info('FOLD %02d | train=%d val=%d batch=%d | v20 ISOLATED D03 RESIDUAL SBP (absolute backbone isolated; DBP/RIG v14)',k,len(trds),len(vads),batch)
    for ep in range(start,EPOCHS+1):
        tl,tm,_,_,_=run_epoch(model,tr,crit,opt,device,sc,True)
        vl,vm,rows,segrows,recrows=run_epoch(model,va,crit,None,device,sc,False,True)
        row={'fold':k,'epoch':ep,'lr':opt.param_groups[0]['lr'],'train_loss':tl,'val_loss':vl,
             **{f'train_{q}':v for q,v in tm.items()},**{f'val_{q}':v for q,v in vm.items()}}
        selection_score(row,sc)
        sch.step(row['val_selection_score'])
        hist.append(row); write_csv(d/'training_history.csv',hist)

        log.info(
            'Fold %02d | %03d/%03d | sel %.4f abs %.4f delta %.4f | '
            'SBP MAE %.2f r %.3f CCC %.3f stdR %.3f dMAE %.2f dR %.3f AUXdMAE %.2f AUXdR %.3f AUXstdR %.3f | '
            'DBP MAE %.2f r %.3f CCC %.3f stdR %.3f dMAE %.2f dR %.3f | '
            'RIG MAE %.4f r %.3f CCC %.3f stdR %.3f dMAE %.4f dR %.3f',
            k,ep,EPOCHS,row['val_selection_score'],row['val_bp_error_score'],row['val_delta_error_score'],
            vm['sbp_mae'],vm['sbp_pearson_r'],vm['sbp_ccc'],vm['sbp_std_ratio'],vm['sbp_delta_mae'],vm['sbp_delta_r'],vm['sbp_aux_delta_mae'],vm['sbp_aux_delta_r'],vm['sbp_aux_delta_std_ratio'],
            vm['dbp_mae'],vm['dbp_pearson_r'],vm['dbp_ccc'],vm['dbp_std_ratio'],vm['dbp_delta_mae'],vm['dbp_delta_r'],
            vm['stiffness_mae'],vm['stiffness_pearson_r'],vm['stiffness_ccc'],vm['stiffness_std_ratio'],vm['stiffness_delta_mae'],vm['stiffness_delta_r'])

        if better(row,best,sc):
            best=dict(row); stale=0
            save_ckpt(ck/'best.pt',model,opt,sch,ep,row,sc,k,contract,best,stale)
            write_csv(d/'best_val_subject_state_predictions.csv',rows)
            write_csv(d/'best_val_recording_predictions.csv',recrows)
            write_csv(d/'best_val_segment_predictions.csv',segrows)
            save_json(d/'best_val_cluster_bootstrap_ci.json',subject_cluster_bootstrap(rows,seed=SEED+k))
            log.info(' -> NEW BEST Stage3-v20-FrozenResidualD03 fold %02d at epoch %d',k,ep)
        else:
            stale+=1
        save_ckpt(ck/'last.pt',model,opt,sch,ep,row,sc,k,contract,best,stale)
        if stale>=PATIENCE:
            log.info('Early stopping fold %02d at epoch %d | best=%d',k,ep,best['epoch']); break

    summary={q:v for q,v in best.items() if q.startswith('val_') or q in ('fold','epoch')}
    summary['best_epoch']=int(best['epoch']); summary['blind_used']=False
    save_json(d/'fold_summary.json',summary); save_json(done,{'contract':contract,'summary':summary})
    del tr,va; gc.collect(); return summary


def main():
    a=parse_args(); seed_all(); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    log,lp=logger_setup(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split,folds,_=load_frozen_protocol(); refs,_,_=load_stage3_manifest()
    devrefs=recordings_for_subjects(refs,split['development_subjects'])
    labels=read_labels_for_allowed_refs(devrefs,split['development_subjects'])
    f1=folds['folds'][0]; sc=TargetScaler().fit(labels,f1['train_subjects'])
    amp_probe=AmpFeatureScaler().fit(CACHE_ROOT/'fold_01',refs,f1['train_subjects'])
    phys_probe=PhysMorphFeatureScaler().fit(CACHE_ROOT/'fold_01',refs,f1['train_subjects'])
    probe=Stage3SubjectStateDataset(CACHE_ROOT/'fold_01',refs,labels,sc,f1['train_subjects'],amp_probe,phys_probe)
    rt=resolve_runtime(probe,device); batch=rt['batch_size']; contract=build_contract(rt)

    log.info('PulseBridge Stage3 v20 Frozen-Residual-D03 | batch=%d',batch)
    log.info('CV scope = %s',CV_SCOPE)
    log.info('Blind = LOCKED / NOT OPENED / NOT EVALUATED.')

    targets=folds['folds'] if a.fold is None else [f for f in folds['folds'] if int(f['fold'])==a.fold]
    summaries=[one_fold(f,refs,labels,batch,device,log,contract,a.resume) for f in targets]
    if a.fold is not None: return

    out=OUTPUT_ROOT/'cross_validation'; out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'cv_summary.csv',summaries)
    eps=[s['best_epoch'] for s in summaries]
    agg={'k':10,'best_epochs':eps,'median_best_epoch':int(round(statistics.median(eps))),
         'blind_used':False,'cv_scope':CV_SCOPE,'stage':V11_VERSION}
    for key in sorted({k for s in summaries for k,v in s.items() if k.startswith('val_') and isinstance(v,(int,float)) and np.isfinite(v)}):
        arr=np.array([s[key] for s in summaries],float)
        agg[key+'_mean']=float(arr.mean()); agg[key+'_std']=float(arr.std(ddof=1))
    save_json(out/'cv_aggregate.json',agg)
    log.info('10-FOLD v20-FROZEN-RESIDUAL-D03 COMPLETE | SBP MAE %.2f r %.3f stdR %.3f dMAE %.2f dR %.3f | '
             'DBP MAE %.2f r %.3f stdR %.3f dMAE %.2f dR %.3f | '
             'RIG MAE %.3f r %.3f stdR %.3f dMAE %.3f dR %.3f',
             agg['val_sbp_mae_mean'],agg['val_sbp_pearson_r_mean'],agg['val_sbp_std_ratio_mean'],agg['val_sbp_delta_mae_mean'],agg['val_sbp_delta_r_mean'],
             agg['val_dbp_mae_mean'],agg['val_dbp_pearson_r_mean'],agg['val_dbp_std_ratio_mean'],agg['val_dbp_delta_mae_mean'],agg['val_dbp_delta_r_mean'],
             agg['val_stiffness_mae_mean'],agg['val_stiffness_pearson_r_mean'],agg['val_stiffness_std_ratio_mean'],agg['val_stiffness_delta_mae_mean'],agg['val_stiffness_delta_r_mean'])


if __name__=='__main__':
    main()
