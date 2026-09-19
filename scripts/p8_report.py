"""P8 的結果表：對主結果 arm 的跨種子配對比較。"""
import sys, glob, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,'scripts'); import icir_gap as ig

SD_Y_TRAIN = 0.012459          # train 逐日橫截面 sd(y)，權重縮放用的那個數

def per_seed(arm):
    out={}
    for d in sorted(glob.glob(f"runs/**/*{arm}_s*", recursive=True)):
        s=d.rsplit("_s",1)[-1]
        if not s.isdigit(): continue
        f=f"{d}/predictions/test_predictions.csv"
        if not glob.glob(f): continue
        df=pd.read_csv(f)
        H=df.pivot(index="target_date",columns="ticker",values="y_hat").sort_index().to_numpy()
        Y=df.pivot(index="target_date",columns="ticker",values="y").sort_index().to_numpy()
        ic=ig.ic_series(H,Y)
        ric=np.array([stats.spearmanr(H[t],Y[t]).statistic if H[t].std()>0 and Y[t].std()>0
                      else np.nan for t in range(len(Y))])
        sh,sy=H.std(1),Y.std(1); ok=sy>1e-15
        out[s]=dict(ic=np.nanmean(ic), ric=np.nanmean(ric),
                    icir=ig.icir(ic), ricir=ig.icir(ric),
                    disp=(sh[ok]/sy[ok]).mean(), sd_hat=sh.mean(),
                    daily_ic=ic, daily_ric=ric)
    return out

ARMS=[("主結果 baseline","tw50_betaF1nA2r1", 1.0),
      ("P8b 平衡保持","tw50_p8zbal", SD_Y_TRAIN),
      ("P8a 權重不動","tw50_p8z", 1.0)]
R={nm:per_seed(a) for nm,a,_ in ARMS}
print(f"{'arm':16s} {'n':>3s} {'IC':>16s} {'RankIC':>16s} {'ICIR':>16s} {'RankICIR':>16s}")
for nm,_,_ in ARMS:
    d=R[nm]
    if not d: print(f"{nm:16s}  (無)"); continue
    f=lambda k: (np.mean([v[k] for v in d.values()]), np.std([v[k] for v in d.values()],ddof=1) if len(d)>1 else np.nan)
    print(f"{nm:16s} {len(d):3d} " + " ".join(f"{m:+8.4f} ±{s:6.4f}" for m,s in (f('ic'),f('ric'),f('icir'),f('ricir'))))

print(f"\n{'arm':16s} {'離散比 vs y':>12s} {'ŷ 的日 sd':>11s} {'訓練目標的 sd':>13s} {'ŷ/目標':>9s}")
for nm,_,tsd in ARMS:
    d=R[nm]
    if not d: continue
    sd_hat=np.mean([v['sd_hat'] for v in d.values()]); disp=np.mean([v['disp'] for v in d.values()])
    tgt = SD_Y_TRAIN if tsd==1.0 and nm.startswith("主") else 1.0
    print(f"{nm:16s} {disp:12.2f} {sd_hat:11.4f} {tgt:13.4f} {sd_hat/tgt:9.2f}")

base=R["主結果 baseline"]
for nm,_,_ in ARMS[1:]:
    d=R[nm]
    if not d or not base: continue
    common=sorted(set(d)&set(base), key=int)
    if len(common)<3: print(f"\n[{nm}] 共同種子只有 {len(common)} 顆，跳過檢定"); continue
    print(f"\n[{nm}] vs 主結果 baseline —— 共同種子 n={len(common)}")
    print(f"  {'指標':>10s} {'差':>9s} {'配對 t p':>10s} {'同號':>7s}")
    for k,lab in (('ic','IC'),('ric','RankIC'),('icir','ICIR'),('ricir','RankICIR'),('disp','離散比')):
        a=np.array([d[s][k] for s in common]); b=np.array([base[s][k] for s in common])
        t,p=stats.ttest_rel(a,b)
        print(f"  {lab:>10s} {np.mean(a-b):+9.4f} {p:10.4f} {int(np.sum(np.sign(a-b)==np.sign(np.mean(a-b)))):3d}/{len(common)}")
    # 逐日配對（HAC）
    da=np.nanmean([d[s]['daily_ic'] for s in common],0); db=np.nanmean([base[s]['daily_ic'] for s in common],0)
    ok=np.isfinite(da)&np.isfinite(db); dd=da[ok]-db[ok]
    n=len(dd); L=int(4*(n/100)**(2/9)); s2=np.var(dd,ddof=1)
    for l in range(1,L+1): s2+=2*(1-l/(L+1))*np.cov(dd[l:],dd[:-l],ddof=1)[0,1]
    tt=dd.mean()/(s2/n)**0.5
    print(f"  逐日 IC HAC：Δ {dd.mean():+.4f}  p {2*(1-stats.norm.cdf(abs(tt))):.4f}  (lag={L})")
