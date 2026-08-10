"""扫描简单集成的 holdout R²，短list候选(供4个提交名额挑)。

用 oof_all.pkl 的干净 OOF，测各种"手设权重"简单集成(LGBM+CB+NN，无XGB)的 holdout R²。
holdout 绝对值不靠谱(regime差)，但同类候选的相对排序有参考。
"""
from __future__ import annotations
import pickle
import numpy as np

def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)

def main():
    d=pickle.load(open("/mnt/iscsi/hd/xxz/runs/oof_all.pkl","rb"))
    keys=d["keys"]; oofs=d["oofs"]; yv=d["yv"]; wv=d["wv"]
    print("models:",keys, flush=True)
    P={k:oofs[k] for k in keys}
    # NN 去均值对齐 lgb
    lm=P["lgb"].mean()
    for k in keys:
        if k.startswith("nn"): P[k]=P[k]-P[k].mean()+lm
    # 单模型基线
    print("\n单模型 holdout:", {k:f"{wr2(yv,P[k],wv):+.5f}" for k in keys}, flush=True)
    print("\n=== 简单集成扫描(无XGB) ===", flush=True)
    results=[]
    # LGBM+CB+nn1_10s, 各种NN权重
    for nw in [0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50]:
        gw=(1-nw)/2
        r=wr2(yv, gw*P["lgb"]+gw*P["cb"]+nw*P["nn1"], wv)
        results.append((f"lcb+nn1(10s) nn={nw:.2f}", r))
    # LGBM+CB+nn3_10s
    for nw in [0.20,0.25,0.30,0.35,0.40,0.45,0.50]:
        gw=(1-nw)/2
        r=wr2(yv, gw*P["lgb"]+gw*P["cb"]+nw*P["nn3_emb32"], wv)
        results.append((f"lcb+nn3(10s) nn={nw:.2f}", r))
    # LGBM+CB+nn1+nn3 (两NN)
    for nnw in [0.30,0.40,0.50,0.60]:
        gw=(1-nnw)/2; hw=nnw/2
        r=wr2(yv, gw*P["lgb"]+gw*P["cb"]+hw*P["nn1"]+hw*P["nn3_emb32"], wv)
        results.append((f"lcb+nn1+nn3 nn_total={nnw:.2f}", r))
    # LGBM+CB+nn1+nn2+nn3 (三NN)
    for nnw in [0.40,0.50,0.60]:
        gw=(1-nnw)/2; hw=nnw/3
        r=wr2(yv, gw*P["lgb"]+gw*P["cb"]+hw*P["nn1"]+hw*P["nn2"]+hw*P["nn3_emb32"], wv)
        results.append((f"lcb+nn1+nn2+nn3 nn_total={nnw:.2f}", r))
    results.sort(key=lambda x:-x[1])
    print("\nTop 15 (holdout R²):", flush=True)
    for n,r in results[:15]:
        print(f"  {r:+.5f}  {n}", flush=True)


if __name__=="__main__":
    main()
