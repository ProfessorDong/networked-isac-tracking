"""
Experiment A: belief-state MPC beamforming (Theorem 4) — convergence of the
certified proximal SCA-WMMSE and closed-loop rate/tracking trade-off.

Scene: B=3 BSs (M=16 ULA, K=2 single-antenna users each), two moving targets.
A building (known map) blocks target 1 from BS0 and BS2 while it crosses
x in [90, 130] m, so only BS1 can sense it there.

Methods (Part A2: all use the same centralised EKF; only the beamformer differs):
  mpcA / mpcD       H=4 belief-state MPC, A-criterion (-log tr of the position
                    PCRB) or D-criterion (log det) over the predicted recursion
  myopicA / myopicD H=1 with the tracker prior (per-slot design of the
                    [ParetoISAC] type, which uses the D-criterion)
  snapshot / snapshotA  H=1 without the prior (snapshot design, D- or A-criterion;
                    both use J = 1e-6 I + J_obs, so either is defined even where the
                    snapshot information is singular)
  comm              eta=1 (communication-only WMMSE)
Part A3 (integrated loop): the same controller acts on the DISTRIBUTED belief of BS0
(the controller location): every BS runs its own EKF prediction/linearisation, the
three BSs fuse anchored HCMCI messages by I rounds of compensating consensus over a
line backhaul (BS0-BS1-BS2) with m-bit pow2-range quantization and symmetric link
failures; beams for slot t are designed from the beliefs after slot t-1 (one-slot
feedback delay). Channels, target paths and measurement-noise draws are shared.
Measurement noise is pre-drawn per (t, target, BS) regardless of visibility/SNR
(common random numbers); targets move with the piecewise-constant-acceleration
model whose covariance is cv_Q (matched to the filter).
Outputs results/exp_A.json.
"""
import sys, json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import isac_core as ic
from isac_core import cv_F, cv_Q, h_mono, H_mono, wrap, SIG_REF, SNR_MAX
from isac_mpc import (Scene, steer, build_horizon, solve, mrt_init, SolveLog,
                      comm, objective, gains)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_A.json"
DT, T = 0.1, 130
SIGMA_A = [0.5, 6.0]     # slow vehicle, manoeuvring target
BS_POS = np.array([[0, 0], [200, 0], [100, 173.0]])
BLOCK_X = (90.0, 130.0)


def blocked(b, xy):
    # building south of the road segment shadows target 1 from BS0 and BS2
    return (b in (0, 2)) and (BLOCK_X[0] <= xy[0] <= BLOCK_X[1]) and xy[1] < 70


def make_scene(seed, M=16, K=2, snr_ref=1.0):
    rng = np.random.default_rng(seed)
    B = len(BS_POS)
    cen = BS_POS.mean(0)
    az = np.array([np.arctan2(*(cen - p)[::-1]) for p in BS_POS])
    users = np.zeros((B, K, 2))
    for b in range(B):
        for k in range(K):
            ang = az[b] + rng.uniform(-1.0, 1.0)
            users[b, k] = BS_POS[b] + rng.uniform(40, 90) * np.array([np.cos(ang), np.sin(ang)])
    h = np.zeros((B, B, K, M), complex)
    for bt in range(B):
        for bu in range(B):
            for k in range(K):
                d = users[bu, k] - BS_POS[bt]
                r = np.linalg.norm(d)
                phi = np.arctan2(d[1], d[0]) - az[bt]
                beta, Kr = 10 * (100 / r) ** 3, 10.0
                nlos = (rng.normal(size=M) + 1j * rng.normal(size=M)) / np.sqrt(2)
                h[bt, bu, k] = np.sqrt(beta) * (np.sqrt(Kr / (Kr + 1)) * np.sqrt(M) * steer(M, phi)
                                                + np.sqrt(1 / (Kr + 1)) * nlos)
    return Scene(BS_POS, az, M, K, h, alpha=np.ones((B, K)), snr_ref=snr_ref, blocked=blocked)


def make_targets(seed):
    rng = np.random.default_rng(100 + seed)
    F = cv_F(DT)
    x1 = np.array([30.0, 40.0 + rng.normal(0, 2), 12.0, 0.0])
    x2 = np.array([160.0, 130.0 + rng.normal(0, 2), -6.0, -4.0])
    G = np.vstack([DT ** 2 / 2 * np.eye(2), DT * np.eye(2)])    # Q = sigma_a^2 G G^T = cv_Q
    trajs = []
    for x, sa in zip((x1, x2), SIGMA_A):
        tr = [x.copy()]
        for _ in range(T - 1):
            x = F @ x + G @ rng.normal(0, sa, 2)
            tr.append(x.copy())
        trajs.append(np.array(tr))
    return trajs


def measure(sc, w0, x, mu, b, noise):
    """Monostatic report of BS b with the applied beams at the true state x (or None);
    the tracker evaluates the noise level at its own prediction mu."""
    if blocked(b, x[:2]):
        return None
    a = sc.steer_to(b, x[:2])
    g = (sc.M / sc.P) * np.sum(np.abs(a.conj() @ w0[0, b].T) ** 2)
    r = max(np.linalg.norm(x[:2] - BS_POS[b]), 1.0)
    snr = min(sc.snr_ref * (100.0 / r) ** 4 * g, SNR_MAX)
    if snr < 1e-2:
        return None
    z = h_mono(x, BS_POS[b]) + noise * np.sqrt(SIG_REF ** 2 / snr)
    ap = sc.steer_to(b, mu[:2])
    gp = (sc.M / sc.P) * np.sum(np.abs(ap.conj() @ w0[0, b].T) ** 2)
    rp = max(np.linalg.norm(mu[:2] - BS_POS[b]), 1.0)
    Ru = SIG_REF ** 2 / max(min(sc.snr_ref * (100.0 / rp) ** 4 * gp, SNR_MAX), 1e-2)
    Hm = H_mono(mu, BS_POS[b])
    inn = z - h_mono(mu, BS_POS[b])
    inn[0] = wrap(inn[0])
    return Hm.T @ np.diag(1 / Ru) @ Hm, Hm.T @ np.diag(1 / Ru) @ inn


def closed_loop(args):
    """args = (method, eta, seed, H[, fus]) with fus = None (centralised EKF) or
    dict(I=, bits=, p=) for the distributed belief of BS0 (Part A3)."""
    method, eta, seed, H = args[:4]
    fus = args[4] if len(args) > 4 else None
    sc = make_scene(seed)
    trajs = make_targets(seed)
    rng = np.random.default_rng(500 + seed)
    noise = np.random.default_rng(700 + seed).normal(size=(T, len(trajs), len(BS_POS), 3))
    F = cv_F(DT)
    Qs = [cv_Q(DT, s) for s in SIGMA_A]
    mus = [tr[0] + rng.normal(0, 1, 4) * np.array([2, 2, 1, 1]) for tr in trajs]
    Ps = [np.diag([4.0, 4.0, 1.0, 1.0]) for _ in trajs]
    nB = len(BS_POS)
    if fus is not None:                          # per-BS beliefs, common initialisation
        mb = [[m.copy() for m in mus] for _ in range(nB)]
        Pb = [[P.copy() for P in Ps] for _ in range(nB)]
        Wc = ic.metropolis(ic.line_adj(nB))
        rq = np.random.default_rng(900 + seed)
        nfail = 0
    Hh = 1 if method.startswith(("myopic", "snapshot", "comm")) else H
    crit = "A" if method.endswith("A") else "D"
    eta_m = 1.0 if method == "comm" else eta
    W = mrt_init(sc, Hh)
    err2 = np.zeros((T, len(trajs)))
    rate = np.zeros(T)
    for t in range(T):
        bel_mu = mus if fus is None else mb[0]   # controller at BS0 uses BS0's belief
        bel_P = Ps if fus is None else Pb[0]
        hz = build_horizon(sc, bel_mu, bel_P, DT, SIGMA_A, Hh,
                           prior_free=method.startswith("snapshot"), crit=crit)
        W, _ = solve(sc, hz, W, eta_m, n_iter=40 if t else 150, tol=1e-6)
        w0 = W[:1]
        _, sinr = comm(sc, w0)
        rate[t] = float(np.sum(np.log2(1 + sinr)))
        for q, tr in enumerate(trajs):
            x = tr[t]
            if fus is None:
                mu = F @ mus[q]
                P = F @ Ps[q] @ F.T + Qs[q]
                Lam, nu = np.linalg.inv(P), np.zeros(4)
                for b in range(nB):
                    m = measure(sc, w0, x, mu, b, noise[t, q, b])
                    if m is not None:
                        Lam, nu = Lam + m[0], nu + m[1]
                Ps[q] = np.linalg.inv(Lam)
                mus[q] = mu + Ps[q] @ nu
                e = mus[q][:2] - x[:2]
            else:
                pm = [F @ mb[b][q] for b in range(nB)]
                pP = [F @ Pb[b][q] @ F.T + Qs[q] for b in range(nB)]
                infos = []
                for b in range(nB):
                    m = measure(sc, w0, x, pm[b], b, noise[t, q, b])
                    infos.append(m if m is not None else (np.zeros((4, 4)), np.zeros(4)))
                c = pm[0].copy()                 # anchor: BS0's prediction (controller BS)
                newm, newP, nf = ic.dist_fuse(pm, pP, infos, c, Wc, fus["I"], fus["bits"],
                                              fus["p"], rq, rule="compensating", fusion="hcmci",
                                              range_code="pow2")
                nfail += nf
                for b in range(nB):
                    mb[b][q], Pb[b][q] = newm[b], newP[b]
                e = mb[0][q][:2] - x[:2]
            err2[t, q] = e @ e
        if Hh > 1:
            W = np.concatenate([W[1:], W[-1:]], 0)
    out = dict(method=method, eta=eta, seed=seed, H=Hh, err2=err2.tolist(), rate=rate.tolist())
    if fus is not None:
        out.update(fus=fus, spdfail=int(nfail))
    return out


def convergence_runs():
    """Part A1: certified-rho convergence traces on one instance."""
    sc = make_scene(0)
    trajs = make_targets(0)
    mus = [tr[60] for tr in trajs]
    Ps = [np.diag([1.0, 1.0, 0.5, 0.5]) for _ in trajs]
    out = {}
    for crit, H in (("D", 1), ("D", 4), ("A", 4)):
        hz = build_horizon(sc, mus, Ps, DT, SIGMA_A, H, crit=crit)
        for eta in (0.2, 0.5, 0.8):
            lg = SolveLog()
            Wn, f = solve(sc, hz, mrt_init(sc, H), eta, n_iter=400, tol=0.0, log=lg)
            out[f"{crit}_H{H}_eta{eta}"] = dict(f=lg.f, rho=lg.rho, resid=lg.resid, trials=lg.trials)
            print(f"A1 {crit} H={H} eta={eta}: f {lg.f[0]:.3f} -> {lg.f[-1]:.3f}, "
                  f"min step {np.min(np.diff(lg.f)):.2e}, resid {lg.resid[-1]:.2e}, "
                  f"backtracks {sum(t - 1 for t in lg.trials)}", flush=True)
    return out


def main(quick=False):
    seeds = range(2 if quick else 20)
    etas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    jobs = []
    for s in seeds:
        for eta in etas:
            for m in ("mpcA", "myopicA", "mpcD", "myopicD", "snapshot", "snapshotA"):
                jobs.append((m, eta, s, 4))
        jobs.append(("comm", 1.0, s, 1))
    # Part A3: integrated distributed loop (controller at BS0), A-criterion
    FUS = {"full_I2": dict(I=2, bits=None, p=0.0), "m8_I2": dict(I=2, bits=8, p=0.0),
           "m8_I2_p02": dict(I=2, bits=8, p=0.2), "m4_I2_p02": dict(I=2, bits=4, p=0.2),
           "m8_I1": dict(I=1, bits=8, p=0.0)}
    for s in seeds:
        for eta in (0.2, 0.6):
            for m in ("mpcA", "myopicA"):
                for nm, fz in FUS.items():
                    jobs.append((m, eta, s, 4, dict(fz, name=nm)))
    conv = convergence_runs()
    with Pool(min(18, len(jobs))) as pool:
        runs = pool.map(closed_loop, jobs)
    summary = {}
    for r in runs:
        e = np.array(r["err2"])
        key = f"{r['method']}_{r['eta']}" + (f"_{r['fus']['name']}" if "fus" in r else "")
        d = summary.setdefault(key, dict(method=r["method"], eta=r["eta"], rmse=[],
                                         rmse_block=[], rate=[]))
        d["rmse"].append(float(np.sqrt(e[10:].mean())))
        tr1 = make_targets(r["seed"])[0]
        inb = (tr1[:, 0] >= BLOCK_X[0]) & (tr1[:, 0] <= BLOCK_X[1])
        d["rmse_block"].append(float(np.sqrt(e[inb, 0].mean())))
        d["rate"].append(float(np.mean(r["rate"])))
    for k, d in summary.items():
        print(f"{k:16s} rmse {np.mean(d['rmse']):.3f}  block {np.mean(d['rmse_block']):.3f}  "
              f"rate {np.mean(d['rate']):.2f}", flush=True)
    if not quick:
        OUT.write_text(json.dumps(dict(convergence=conv, summary=summary,
                                       runs=[{k: v for k, v in r.items()} for r in runs],
                                       block_x=BLOCK_X, T=T, dt=DT)))
        print("wrote", OUT)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
