"""用早段 holdout（似公榜）优化权重，避免 ratio 过拟合。"""
import pickle, numpy as np, pandas as pd, json
from pathlib import Path
from scipy.optimize import minimize

def load(p): return pickle.load(open(p,'rb'))
def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)

RUN=Path('/mnt/iscsi/hd/xxz/runs'); SUB=Path('/mnt/iscsi/hd/xxz/submissions')
d1=pickle.load(open(RUN/'oof_all.pkl','rb')); oofs=dict(d1['oofs']); yv=d1['yv']; wv=d1['wv']; tids=d1['tids_holdout']
d2=pickle.load(open(RUN/'ratio4_oof.pkl','rb'))
for k,v in d2['oofs'].items(): oofs[k]=v
d3=pickle.load(open(RUN/'ratio_v2_oof.pkl','rb'))
for k,v in d3['oofs'].items(): oofs[k]=v
d4=pickle.load(open(RUN/'ratio_sum_oof.pkl','rb'))
for k,v in d4['oofs'].items(): oofs[k]=v

unique_tids=np.unique(tids); n=len(unique_tids)
early_tids = unique_tids[:n//2]
early_mask = np.isin(tids, early_tids)
yv_e, wv_e = yv[early_mask], wv[early_mask]
print(f'early holdout: {early_mask.sum()} rows, {len(early_tids)} tids', flush=True)

candidates_sets = {
    'A_raw_only': ['lgb','cb','xgb','nn3_emb32'],
    'B_ens_ratio_style': ['ratio_lgb','cb','nn3_emb32'],
    'C_ratio_lgb_plus_raw': ['ratio_lgb','cb','xgb','nn3_emb32'],
    'D_all_raw_plus_rlgb': ['lgb','ratio_lgb','cb','xgb','nn3_emb32','nn1'],
    'E_no_ratio_cb': ['lgb','ratio_lgb','cb','xgb','nn3_emb32','nn1','ratio_sum_lgb'],
    'F_champ_plus_xgb': ['ratio_lgb','cb','nn3_emb32','xgb'],
    'G_simple_lcb_nn': ['lgb','cb','nn3_emb32'],
    'H_rlgb_cb_nn_xgb_nn1': ['ratio_lgb','cb','xgb','nn1','nn3_emb32'],
}

TE_FILE = {
    'lgb':'lgbm_full_submission.csv','cb':'cb_submission.csv','xgb':'xgb_submission.csv',
    'nn1':'nn1_10s.csv','nn3_emb32':'nn3_10s.csv',
    'ratio_lgb':'ratio4_ratio_lgb.csv','ratio_cb':'ratio4_ratio_cb.csv',
    'ratio_cb_tuned':'v2_ratio_cb_tuned.csv','ratio_sum_lgb':'rsum_ratio_sum_lgb.csv',
}
base = pd.read_csv(SUB/'lgbm_full_submission.csv').sort_values('row_id').reset_index(drop=True)

print('\n=== 早段优化权重候选 ===', flush=True)
results = []
for name, keys in candidates_sets.items():
    P = np.array([oofs[k] for k in keys])
    P_e = P[:, early_mask]
    def neg(w): return -wr2(yv_e, w @ P_e, wv_e)
    cons=({'type':'eq','fun':lambda w:w.sum()-1}); bnds=[(0,1)]*len(keys)
    best=None; np.random.seed(2026)
    for _ in range(30):
        r=minimize(neg, np.random.dirichlet(np.ones(len(keys))), method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter':300,'ftol':1e-9})
        if best is None or r.fun<best.fun: best=r
    w=np.maximum(best.x,0); w=w/w.sum()
    early_r2 = wr2(yv_e, w @ P_e, wv_e)
    full_r2 = wr2(yv, w @ P, wv)
    ws = ', '.join(f'{k}:{wi:.2f}' for k,wi in zip(keys,w) if wi>0.01)
    print(f"  {name}: 早={early_r2:+.6f} 全={full_r2:+.6f}  [{ws}]", flush=True)
    results.append((name, keys, w, early_r2, full_r2))

results.sort(key=lambda x: -x[3])
print(f"\n=== 写 top 5 早段最优候选 ===", flush=True)
for name, keys, w, er2, fr2 in results[:5]:
    T=[]; vk=[]
    for k in keys:
        d=pd.read_csv(SUB/TE_FILE[k]).sort_values('row_id').reset_index(drop=True)
        T.append(d['target'].to_numpy(np.float32)); vk.append(k)
    T=np.array(T); w_v=np.array([w[keys.index(k)] for k in vk]); w_v=w_v/w_v.sum()
    lgb_keys=[k for k in vk if 'lgb' in k]
    ref=np.mean([T[vk.index(k)].mean() for k in lgb_keys]) if lgb_keys else T[0].mean()
    for i,k in enumerate(vk):
        if 'nn' in k and 'lgb' not in k: T[i]=T[i]-T[i].mean()+ref
    pred=w_v@T; pred=np.where(np.isfinite(pred),pred,0.0)
    out=SUB/f'early_ens_{name}.csv'
    pd.DataFrame({'row_id':base['row_id'],'target':pred}).to_csv(out,index=False)
    print(f"  → {out.name} 早={er2:+.6f} 全={fr2:+.6f} mean={pred.mean():+.5f}", flush=True)
