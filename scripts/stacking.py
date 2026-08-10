"""stacking 元学习器：用 LGBM 在 OOF 预测上做非线性混合，看能否超线性 blend。

holdout 按时间分两半: meta-train(前半) / meta-eval(后半)。在 meta-train 上训 LGBM 元模型，
在 meta-eval 上对比线性 blend。若元模型明显胜出，则用全 holdout 训元模型 → 套用 test。
"""
from __future__ import annotations
import pickle, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

def wr2(y,p,w):
    d=float(np.sum(w*y*y)); return 0.0 if d<=0 else 1-float(np.sum(w*(y-p)**2)/d)

def main():
    d=pickle.load(open("/mnt/iscsi/hd/xxz/runs/oof_all.pkl","rb"))
    keys=d["keys"]; oofs=d["oofs"]; yv=d["yv"]; wv=d["wv"]; tids=d["tids_holdout"]
    P=np.array([oofs[k] for k in keys])  # [M, N]
    M,N=P.shape
    print("models:",keys, "holdout N=",N, flush=True)
    # 按时间排序分半
    order=np.argsort(tids); half=N//2
    tr_idx=order[:half]; ev_idx=order[half:]
    Ptr=P[:,tr_idx]; Pev=P[:,ev_idx]; ytr=yv[tr_idx]; yev=yv[ev_idx]; wtr=wv[tr_idx]; wev=wv[ev_idx]
    # 线性 blend 对照(在 meta-train 优化, meta-eval 评)
    from scipy.optimize import minimize
    def neg(w): return -wr2(ytr, w@Ptr, wtr)
    cons={"type":"eq","fun":lambda w:w.sum()-1}; bnds=[(0,1)]*M; best=None
    for _ in range(30):
        r=minimize(neg,np.random.dirichlet(np.ones(M)),method="SLSQP",bounds=bnds,constraints=cons)
        if best is None or r.fun<best.fun: best=r
    wl=best.x; wl=np.maximum(wl,0); wl/=wl.sum()
    r2_lin=wr2(yev, wl@Pev, wev)
    print(f"线性 blend meta-eval R²={r2_lin:+.5f} weights={[round(float(x),2) for x in wl]}", flush=True)
    # LGBM 元模型
    dtr=lgb.Dataset(Ptr.T,label=ytr,weight=wtr,free_raw_data=False)
    dva=lgb.Dataset(Pev.T,label=yev,weight=wev,reference=dtr,free_raw_data=False)
    def feval(p,ds):
        yy=ds.get_label(); ww=ds.get_weight() or np.ones_like(yy)
        dd=float(np.sum(ww*yy*yy)); s=0 if dd<=0 else 1-float(np.sum(ww*(yy-p)**2)/dd)
        return ("wr2",float(s),True)
    mm=lgb.train(dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=15,min_data_in_leaf=5000,
                      feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbosity=-1,num_threads=32,seed=2026),
                 dtr,num_boost_round=200,valid_sets=[dva],valid_names=["va"],feval=feval,
                 callbacks=[lgb.early_stopping(30,verbose=False),lgb.log_evaluation(0)])
    r2_meta=wr2(yev, mm.predict(Pev.T), wev)
    print(f"LGBM 元模型 meta-eval R²={r2_meta:+.5f} (Δ vs 线性 {r2_meta-r2_lin:+.5f})", flush=True)
    print(f"  best_iter={mm.best_iteration}", flush=True)
    # 若元模型胜出 >+0.00003，用全 holdout 训元模型并套用 test
    if r2_meta - r2_lin > 3e-5:
        print("\n元模型胜出，生成 stacking 提交...", flush=True)
        Pall=lgb.Dataset(P.T,label=yv,weight=wv,free_raw_data=False)
        bi=mm.best_iteration or 100
        mfull=lgb.train(dict(objective="regression",metric="None",learning_rate=0.05,num_leaves=15,min_data_in_leaf=5000,
                          feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5,verbosity=-1,num_threads=32,seed=2026),
                       Pall,num_boost_round=bi,feval=feval,callbacks=[lgb.log_evaluation(0)])
        S=Path("/mnt/iscsi/hd/xxz/submissions")
        tmap={"lgb":"lgbm_full_submission","cb":"cb_submission","xgb":"xgb_submission","nn1":"nn1_10s","nn2":"nn2_submission","nn3_emb32":"nn3_10s","nn4_deep4":"nnvar_deep4"}
        T=np.array([pd.read_csv(S/f"{tmap[k]}.csv").sort_values("row_id")["target"].to_numpy() for k in keys])
        lgb_mean=T[0].mean()
        for i,k in enumerate(keys):
            if k.startswith("nn"): T[i]=T[i]-T[i].mean()+lgb_mean
        pred=mfull.predict(T.T); pred=np.where(np.isfinite(pred),pred,0.0)
        base=pd.read_csv(S/"lgbm_full_submission.csv").sort_values("row_id").reset_index(drop=True)
        pd.DataFrame({"row_id":base.row_id,"target":pred}).to_csv(S/"ensemble_stacking.csv",index=False)
        print(f"wrote ensemble_stacking.csv mean={pred.mean():+.4f}", flush=True)
    else:
        print("\n元模型未明显胜出，维持线性 ensemble_opt", flush=True)

if __name__=="__main__":
    import numpy as np
    main()
