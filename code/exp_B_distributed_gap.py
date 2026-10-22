"""
Experiment B: quantized, lossy consensus fusion (Theorem 2, Corollary 1).

Proposed tracker: anchored HCMCI messages, compensating consensus update
(average preserved). Ablations: the plain update X(i+1) = C(i) Xhat(i)
(drift of the average) and CM fusion (consensus on new information only).

B = 6 BSs on a 300 m ring around a synthetic CV trajectory. For ring and line
cooperation graphs we sweep the number of consensus rounds I, the bits per
transmitted scalar b (every transmission is requantized with subtractive
dither), and the symmetric link-failure probability p. For every run we log,
per time step, the realised consensus error ||X(I) - 1 s^T||_F^2 and the
right-hand side of Theorem 2 evaluated with the realised quantizer steps.
Tracking RMSE uses common random numbers against the centralised EKF.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from isac_core import (synth_trajectory, ring_bss, ring_adj, line_adj, metropolis,
                       lambda_bar, rho_bar, FusionCfg, run_trackers, rmse)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_B.json"
T, DT, B = 200, 0.1, 6
TRAJ = synth_trajectory(T, DT)
BSS = ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
TOPO = {"ring": ring_adj(B), "line": line_adj(B)}
I_GRID = list(range(1, 41)) + [55, 89]
SEEDS = range(10)


def job(args):
    topo, bits, p, rule, fusion, seed = args
    cfgs = [FusionCfg("cen")] + [FusionCfg(f"I{I}", TOPO[topo], I, bits, p, "anchored", rule, fusion)
                                 for I in I_GRID]
    out, dg = run_trackers(TRAJ, BSS, DT, cfgs, seed=seed, diag=True)
    res = {"cen": rmse(out["cen"]), "cen_nees": float(np.mean(out["cen_nees"][20:])),
           "cen_cov": float(np.mean(out["cen_cov"][20:]))}
    for I in I_GRID:
        d = dg[f"I{I}"]
        a, b = d["act"][20:], d["bnd"][20:]
        ok = b > 0
        res[f"I{I}"] = dict(rmse=rmse(out[f"I{I}"]),
                            nees=float(np.mean(out[f"I{I}_nees"][20:])),
                            cov=float(np.mean(out[f"I{I}_cov"][20:])),
                            act=float(np.mean(a)), bnd=float(np.mean(b)),
                            ratio=float(np.mean(a[ok] / b[ok])) if ok.any() else 0.0,
                            dev0=float(np.mean(d["dev0"][20:])),
                            V=float(np.mean(d["V"][20:])))
    return (topo, bits, p, rule, fusion, seed, res)


def main():
    C, P_, H = "compensating", "plain", "hcmci"
    settings = [(t, None, 0.0, C, H) for t in TOPO] + \
               [("ring", None, 0.0, C, "cm"), ("ring", None, 0.0, C, "hcmci_avg")] + \
               [("ring", b, 0.0, r, H) for b in (4, 6, 8) for r in (C, P_)] + \
               [("line", 8, 0.0, C, H)] + \
               [("ring", 8, p, C, H) for p in (0.2, 0.4)]
    jobs = [(t, b, p, r, f, s) for (t, b, p, r, f) in settings for s in SEEDS]
    with Pool(18) as pool:
        runs = pool.map(job, jobs)
    agg = {}
    keys = ("rmse", "nees", "cov", "act", "bnd", "ratio", "dev0", "V")
    for topo, bits, p, rule, fusion, seed, res in runs:
        key = f"{topo}_b{bits}_p{p}_{rule}_{fusion}"
        a = agg.setdefault(key, dict(topo=topo, bits=bits, p=p, rule=rule, fusion=fusion, cen=[], cen_nees=[], cen_cov=[],
                                     **{f"I{I}": {k: [] for k in keys} for I in I_GRID}))
        a["cen"].append(res["cen"]); a["cen_nees"].append(res["cen_nees"]); a["cen_cov"].append(res["cen_cov"])
        for I in I_GRID:
            for k in keys:
                a[f"I{I}"][k].append(res[f"I{I}"][k])
    meta = {}
    for t, A in TOPO.items():
        W = metropolis(A)
        ev = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
        meta[t] = dict(sigma2=float(ev[1]),
                       lam={str(p): lambda_bar(W, p) for p in (0.0, 0.2, 0.4)},
                       rho={str(p): rho_bar(W, p) for p in (0.0, 0.2, 0.4)})
    for key, a in agg.items():
        line = [f"{key:20s} cen {np.mean(a['cen']):.4f}"]
        for I in (1, 5, 13, 34, 55):
            r = a[f"I{I}"]
            line.append(f"I{I}: {np.median(r['rmse']):.4f} ({np.mean(r['act']):.2e}/{np.mean(r['bnd']):.2e})")
        print(" | ".join(line), flush=True)
    OUT.write_text(json.dumps(dict(I_grid=I_GRID, meta=meta, agg=agg, B=B, T=T)))
    print("wrote", OUT, meta)


if __name__ == "__main__":
    main()
