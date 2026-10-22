"""
Experiment D: message distortion -> tracking deviation (Theorem 3, Corollary 2).

B = 4 BSs, full-mesh cooperation (one exact averaging round), so the only
perturbation is the subtractive-dithered quantizer. We sweep bits per scalar b
for (i) anchored messages (common anchor = leader's predicted mean) and
(ii) absolute-frame messages, with the whole scene translated along x by 0 m,
1 km and 10 km. For each b we compare the empirical excess MSE (distributed minus
centralised, common random numbers) with the Theorem-3 prediction propagated
through the filter by the Lyapunov recursion X_t = A_t X_{t-1} A_t^T + C_t, and
we evaluate the centralized-reference half-step margin gamma_t = ||P*_t|| ||Delta_t||
(a proxy along the centralized path), and log the REALISED margin omega along the actual
quantized filters (run_trackers diagnostics) and any loss of positive definiteness.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from isac_core import (synth_trajectory, ring_bss, full_adj, FusionCfg, run_trackers,
                       rmse, cv_F, cv_Q, gen_measurements, local_info, pack, BLOCKS,
                       IU, N_LAM, BS)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_D.json"
T, DT, B = 200, 0.1, 4
TRAJ = synth_trajectory(T, DT)
BSS = ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
BITS = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24]
ORIGINS = [0.0, 1e3, 1e4]
SEEDS = range(10)


def predicted(seed, bits, form, origin):
    """Theorem-3 prediction along the centralised EKF of the same realisation."""
    rng = np.random.default_rng(1000 + seed)
    sh = np.array([origin, 0.0])
    traj = TRAJ.copy(); traj[:, :2] += sh
    bss = [BS(b.pos + sh, b.snr_ref) for b in BSS]
    Z, Rs = gen_measurements(traj, bss, rng)
    F, Q = cv_F(DT), cv_Q(DT, 1.0)
    rng0 = np.random.default_rng(2000 + seed)
    mu = traj[0] + rng0.normal(size=4) * np.array([2, 2, 1, 1])
    P = np.diag([25.0, 25.0, 4.0, 4.0])
    X = np.zeros((4, 4))
    exc, gam = [], []
    L = 2 ** (bits - 1) - 1                 # same quantizer as isac_core
    for t in range(T):
        mu = F @ mu; P = F @ P @ F.T + Q
        Lpred = np.linalg.inv(P)
        c = np.zeros(4) if form == "absolute" else mu
        msgs, Lsum, nsum = [], np.zeros((4, 4)), np.zeros(4)
        for b in range(B):
            Lb, nb = local_info(mu, Z[t, b], bss[b], Rs[t, b])
            # HCMCI message (prior information / B included), as in run_trackers
            msgs.append(B * pack(Lpred / B + Lb, (Lpred / B + Lb) @ (mu - c) + nb))
            Lsum += Lb; nsum += nb
        msgs = np.array(msgs)
        var = np.zeros(msgs.shape[1]); half = np.zeros(msgs.shape[1])
        for k in range(BLOCKS.max() + 1):
            idx = BLOCKS == k
            step = 1.0001 * np.max(np.abs(msgs[:, idx]), axis=1) / L  # per BS (codec's range factor)
            var[idx] = np.sum(step ** 2 / 12.0) / B ** 2              # fused (mean) error
            half[idx] = np.mean(step) / 2                             # a.s. bound on |error|
        Pst = np.linalg.inv(Lpred + Lsum)
        u = (mu - c) + Pst @ nsum                                     # u = mu* - c (Theorem 3)
        # D_eff = E[(delta - Delta u)(...)^T]
        Deff = np.diag(var[N_LAM:]).astype(float)
        Dlam_abs = np.zeros((4, 4))
        for e, (i, j) in enumerate(zip(*IU)):
            v = np.zeros(4)
            if i == j:
                v[i] = u[i]
            else:
                v[i] = u[j]; v[j] = u[i]
            Deff += var[e] * np.outer(v, v)
            Dlam_abs[i, j] = Dlam_abs[j, i] = half[e]
        C = Pst @ Deff @ Pst
        A = Pst @ Lpred @ F
        X = A @ X @ A.T + C
        exc.append(np.trace(X[:2, :2]))
        gam.append(np.linalg.norm(Pst, 2) * np.linalg.norm(Dlam_abs, 2))
        mu = mu + Pst @ nsum; P = Pst
    return np.array(exc), np.array(gam)


def job(args):
    seed, form, origin = args
    cfgs = [FusionCfg("cen")] + [FusionCfg(f"b{b}", full_adj(B), 1, b, 0.0, form, "plain", "hcmci")
                                 for b in BITS]
    out, dg = run_trackers(TRAJ, BSS, DT, cfgs, seed=seed, origin=np.array([origin, 0.0]),
                           keep_mu=True, diag=True)
    res = {"cen_mse": float(np.mean(out["cen"][20:]))}
    mu_c = out["cen_mu"][20:, 0, :2]
    for b in BITS:
        e = out[f"b{b}"][20:]
        dv = np.sum((out[f"b{b}_mu"][20:, :, :2] - mu_c[:, None]) ** 2, axis=-1).mean()
        ex, gm = predicted(seed, b, form, origin)
        res[f"b{b}"] = dict(mse=float(np.mean(e)) if np.all(np.isfinite(e)) else float("inf"),
                            dev=float(dv) if np.isfinite(dv) else float("inf"),
                            pred_exc=float(np.mean(ex[20:])), gamma_max=float(np.max(gm[20:])),
                            gamma_med=float(np.median(gm[20:])),
                            # realised Theorem-3 margin along the ACTUAL quantized filter
                            # (exact network sum vs. received fused information)
                            omega_max=float(np.nanmax(dg[f"b{b}"]["omega"][20:])),
                            spdfail=int(np.sum(dg[f"b{b}"]["spdfail"])))
    return (seed, form, origin, res)


def main():
    jobs = [(s, "anchored", 0.0) for s in SEEDS] + \
           [(s, "absolute", o) for o in ORIGINS for s in SEEDS] + \
           [(s, "anchored", o) for o in (1e3, 1e4) for s in SEEDS]
    with Pool(18) as pool:
        runs = pool.map(job, jobs)
    agg = {}
    for seed, form, origin, res in runs:
        key = f"{form}_{int(origin)}"
        a = agg.setdefault(key, dict(form=form, origin=origin, cen_mse=[],
                                     **{f"b{b}": dict(mse=[], dev=[], pred_exc=[], gamma_max=[], gamma_med=[],
                                                      omega_max=[], spdfail=[])
                                        for b in BITS}))
        a["cen_mse"].append(res["cen_mse"])
        for b in BITS:
            for k in ("mse", "dev", "pred_exc", "gamma_max", "gamma_med", "omega_max", "spdfail"):
                a[f"b{b}"][k].append(res[f"b{b}"][k])
    for key, a in agg.items():
        cm = np.mean(a["cen_mse"])
        print(key, "cen rmse %.4f" % np.sqrt(cm))
        for b in BITS:
            r = a[f"b{b}"]
            m = np.array(r["mse"])
            print(f"   b={b:2d} dev {np.median(r['dev']):.3e} emp_exc {np.median(m) - cm:.3e} (div {np.sum(~np.isfinite(m) | (m > 1e2))}/"
                  f"{len(m)})  pred {np.mean(r['pred_exc']):.3e}  gamma_max {np.median(r['gamma_max']):.2e}")
    import isac_core as _ic
    OUT.write_text(json.dumps(dict(bits=BITS, agg=agg, provenance=_ic.provenance(__file__)), default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
