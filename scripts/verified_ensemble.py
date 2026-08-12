"""集成已验证候选（避免 holdout 过拟合）。"""
import pandas as pd, numpy as np
from pathlib import Path
S = Path('/mnt/iscsi/hd/xxz/submissions')

verified = {
    'ens_ratio_nn30.csv': 0.00334513,
    'ens_pseudo_nn30.csv': 0.00331999,
    'ensemble_lcb_nn30.csv': 0.00331298,
}

print('=== 已验证候选 ===', flush=True)
preds = {}
for f, score in verified.items():
    d = pd.read_csv(S/f).sort_values('row_id').reset_index(drop=True)
    preds[f] = d['target'].to_numpy(np.float32)
    print(f'  {f}: 公榜={score:.6f} mean={preds[f].mean():+.5f} std={preds[f].std():.5f}', flush=True)

keys = list(preds.keys())
print('\n=== 相关性 ===', flush=True)
for ki in keys:
    row = ' '.join(f'{np.corrcoef(preds[ki],preds[kj])[0,1]:.4f}' for kj in keys)
    print(f'  {ki}: {row}', flush=True)

base = pd.read_csv(S/'ens_ratio_nn30.csv').sort_values('row_id').reset_index(drop=True)

def save(name, p):
    p = np.where(np.isfinite(p), p, 0.0)
    pd.DataFrame({'row_id':base['row_id'],'target':p}).to_csv(S/name, index=False)
    print(f'  → {name} mean={p.mean():+.5f}', flush=True)

print('\n=== 生成集成候选 ===', flush=True)
# 1. 等权 3 候选
save('verified_avg3.csv', np.mean([preds[f] for f in keys], 0))
# 2. 按公榜分数加权
w = np.array([verified[f] for f in keys]); w = w - w.min() + 0.0001; w = w/w.sum()
print(f'  weights: {[f"{k.split(".")[0]}:{wi:.3f}" for k,wi in zip(keys,w)]}', flush=True)
save('verified_weighted.csv', np.average([preds[f] for f in keys], weights=w, axis=0))
# 3. top-2 等权
save('verified_avg_top2.csv', 0.5*preds['ens_ratio_nn30.csv'] + 0.5*preds['ens_pseudo_nn30.csv'])
# 4. 7:3
save('verified_73.csv', 0.7*preds['ens_ratio_nn30.csv'] + 0.3*preds['ens_pseudo_nn30.csv'])
# 5. 6:2:2
save('verified_622.csv', 0.6*preds['ens_ratio_nn30.csv'] + 0.2*preds['ens_pseudo_nn30.csv'] + 0.2*preds['ensemble_lcb_nn30.csv'])
# 6. 8:1:1（最优主导）
save('verified_811.csv', 0.8*preds['ens_ratio_nn30.csv'] + 0.1*preds['ens_pseudo_nn30.csv'] + 0.1*preds['ensemble_lcb_nn30.csv'])

# 加入其他未提交但与已验证候选同思路的（ens_ratio8, ens_ratio15, ens_2lgb_pseudo_nn）
extras = ['ens_ratio8_nn30.csv','ens_ratio15_nn30.csv','ens_2lgb_pseudo_nn.csv','ens_pseudo_nn30.csv']
print('\n=== 扩展已验证风格候选 ===', flush=True)
all_preds = dict(preds)
for f in extras:
    if (S/f).exists():
        d = pd.read_csv(S/f).sort_values('row_id').reset_index(drop=True)
        all_preds[f] = d['target'].to_numpy(np.float32)
        # 与 ens_ratio_nn30 相关性
        c = np.corrcoef(all_preds[f], preds['ens_ratio_nn30.csv'])[0,1]
        print(f'  {f}: corr(ratio_nn30)={c:.4f}', flush=True)

# ens_ratio8 / ens_ratio15 是 ratio_nn30 的变种，加进来增加 ratio 多样性
ratio_family = ['ens_ratio_nn30.csv','ens_ratio8_nn30.csv','ens_ratio15_nn30.csv']
if all((S/f).exists() for f in ratio_family):
    avg_ratio = np.mean([all_preds[f] for f in ratio_family], 0)
    save('ratio_family_avg.csv', avg_ratio)
    # ratio family + pseudo
    save('ratio_family_plus_pseudo.csv', 0.7*avg_ratio + 0.3*all_preds['ens_pseudo_nn30.csv'])
