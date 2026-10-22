"""
Experiment R: two-agent tracking on V2X-Radar-C with REAL 4-D radar detections.

Measurements: for every annotated vehicle, frame and agent, the radar points inside
the vehicle's box (+0.5 m) give [azimuth, range, range-rate] (circular mean of the
point azimuths, median range, median ego-compensated Doppler); see v2x_radar_io.
Association is ORACLE (labels select the points); the values are real. A vehicle
is "detected" by an agent iff at least one radar point falls in its box.

Protocol (fixed before looking at tracking results):
  * Noise model and biases are fitted on the TRAIN split only (robust MAD/median):
      sigma_az(r)^2 = sigma_c^2 / r^2 + sigma_0^2   (extended target: roughly constant
      cross-range spread; fitted on the log standard deviation per range bin),
      sigma_r, sigma_rr constant; per agent.
    Calibration subtracts the train-split median biases.
  * Tracking is evaluated on the held-out VALIDATE and TEST splits.
  * Tracks: longest run >= 6 s in which at least one agent annotates the vehicle
    (as in Exp G); RMSE over all frames after a 10-frame burn-in.
  * sigma_a = 3 m/s^2 is fixed a priori (as Exp G). The inflation factor kappa of
    the fitted R is chosen on the TRAIN split (centralised EKF, tracks detected in
    >= 50 % of frames, median RMSE) and frozen. The full (sigma_a, kappa) grid is
    stored as a sensitivity check: the train RMSE is flat for sigma_a >= 3 because
    the error between detections dominates (single agents cannot observe the
    tangential velocity, Theorem 1).
  * Reported sets: all held-out tracks, and the held-out tracks detected by at
    least one agent in >= 50 % of frames (undetected stretches are prediction only).
  * Every tracker gates each agent's measurement against its own prediction with
    a 3-dof chi-square test at 0.997 (outliers are 7-12 % in azimuth/Doppler).
  * The noise covariance is evaluated at each tracker's own prediction.

Trackers (common random numbers: same init noise and dither stream per seed):
  rsu, ego (single agent); cen (centralised EKF); dist{6,8,10,12} anchored and
  abs{8,12} absolute messages (B=2, one exact averaging round, so quantisation is
  the only perturbation); cen_uncal / dist8_uncal (no bias calibration);
  cen_gauss / dist8_gauss (Gaussian surrogate: same detections, z = h(x) + N(0,R)).
Proposition 1 check: along cen_uncal, the dynamic bias gain T_t applied to the
fitted bias predicts the extra position MSE of the uncalibrated tracker.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import isac_core as ic
import v2x_radar_io as io

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_R.json"
DT = io.DT
MIN_LEN = 60
SEEDS = range(5)
SIGMA_A = 3.0                         # a priori (urban vehicles; as Exp G)
SIGMA_A_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]   # sensitivity only
KAPPA_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DET_MIN = 0.5
GATE = 14.16                          # chi2(3) 0.997 quantile
BURN = 10
R_BINS = [0, 15, 30, 50, 80, 150]


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


# ------------------------------------------------------------------ extraction
def scene_job(sdir):
    """Residual rows (for the noise fit) and tracks with real measurements."""
    sc = io.load_scene(sdir)
    T = sc["T"]
    L = [io.labels_at(sc, t) for t in range(T)]
    ids = set().union(*[set(l) for l in L])
    pts = [{a: io.radar_points(sc, a, t) for a in io.AGENTS} for t in range(T)]
    resid, tracks = [], []
    for vid in ids:
        ok = np.array([vid in L[t] for t in range(T)])
        best, cur, start, bs = 0, 0, 0, 0
        for t in range(T):
            cur = cur + 1 if ok[t] else 0
            if cur == 1:
                start = t
            if cur > best:
                best, bs = cur, start
        if best < 3:
            continue
        sl = range(bs, bs + best)
        p = np.array([L[t][vid][1][:2] for t in sl])
        k = np.ones(3) / 3
        ps = np.column_stack([np.convolve(np.pad(p[:, i], 1, mode="edge"), k, "valid") for i in range(2)])
        v = np.gradient(ps, DT, axis=0)
        traj = np.column_stack([ps, v])
        Z = np.full((best, 2, 3), np.nan)
        npts = np.zeros((best, 2), int)
        sens = np.zeros((best, 2, 2))
        for i, t in enumerate(sl):
            _, loc, ext, yw = L[t][vid]
            for j, a in enumerate(io.AGENTS):
                s = sc["radar_pos"][a][t]
                sens[i, j] = s
                z, n = io.measure(*pts[t][a], s, loc, ext, yw)
                npts[i, j] = n
                if z is not None:
                    Z[i, j] = z
                    if 0 < i < best - 1:          # interior frames: velocity well defined
                        h = ic.h_mono(traj[i], s)
                        resid.append((j, h[1], ic.wrap(z[0] - h[0]), z[1] - h[1], z[2] - h[2]))
        if best >= MIN_LEN:
            tracks.append(dict(scene=sdir.name, split=sdir.parent.name, vid=int(vid),
                               traj=traj, Z=Z, npts=npts, sens=sens))
    return sdir.parent.name, resid, tracks


def fit_noise(resid):
    """Per agent: sigma_c, sigma_0 (azimuth), sigma_r, sigma_rr, and median biases."""
    R = np.array(resid, float)
    model = {}
    for j in (0, 1):
        A = R[R[:, 0] == j]
        r, eaz, er, err = A[:, 1], A[:, 2], A[:, 3], A[:, 4]
        rows = []
        for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
            m = (r > lo) & (r <= hi)
            if m.sum() >= 50:
                rows.append((np.median(r[m]), mad(eaz[m]) ** 2, int(m.sum())))
        rows = np.array(rows)
        # sd_az^2 = sigma_c^2 / r^2 + sigma_0^2, fitted by weighted least squares on the
        # LOG standard deviation (relative error; weights: counts), so that no single
        # range bin dominates the fit
        from scipy.optimize import least_squares
        sd, rr, w = np.sqrt(rows[:, 1]), rows[:, 0], np.sqrt(rows[:, 2])
        res = least_squares(lambda th: w * (0.5 * np.log(np.exp(2 * th[0]) / rr ** 2 + np.exp(2 * th[1]))
                                            - np.log(sd)), x0=[0.0, np.log(1e-3)])
        coef = np.exp(2 * res.x)
        model[j] = dict(sigma_c=float(np.sqrt(coef[0])), sigma_0=float(np.sqrt(coef[1])),
                        sigma_r=float(mad(er)), sigma_rr=float(mad(err)),
                        bias=[float(np.median(eaz)), float(np.median(er)), float(np.median(err))],
                        bins=rows.tolist(), n=int(len(A)))
    return model


# ------------------------------------------------------------------ tracking
def R_of(mod, mu, s, kappa=1.0):
    r = max(np.hypot(mu[0] - s[0], mu[1] - s[1]), 1.0)
    return kappa * np.diag([mod["sigma_c"] ** 2 / r ** 2 + mod["sigma_0"] ** 2, mod["sigma_r"] ** 2, mod["sigma_rr"] ** 2])


def info(mu, P, z, s, mod, kappa):
    """(Lambda, nu, H, Rinv, accepted) at the prediction mu; chi-square gated."""
    if np.any(np.isnan(z)):
        return None
    R = R_of(mod, mu, s, kappa)
    H = ic.H_mono(mu, s)
    innov = z - ic.h_mono(mu, s)
    innov[0] = ic.wrap(innov[0])
    S = H @ P @ H.T + R
    if innov @ np.linalg.solve(S, innov) > GATE:
        return None
    Ri = np.linalg.inv(R)
    return H.T @ Ri @ H, H.T @ Ri @ innov, H, Ri


def run_track(tr, model, sigma_a, kappa, only=None):
    traj, Zr, sens = tr["traj"], tr["Z"], tr["sens"]
    T = len(traj)
    F, Q = ic.cv_F(DT), ic.cv_Q(DT, sigma_a)
    bias = {j: np.array(model[j]["bias"]) for j in (0, 1)}
    S2 = np.eye(4)[:2]
    out = {}
    for seed in SEEDS:
        rng0 = np.random.default_rng(2000 + seed)
        mu0 = traj[0] + rng0.normal(size=4)
        P0 = np.diag([4.0, 4.0, 4.0, 4.0])
        # Gaussian surrogate measurements: same detections, fitted noise at the truth
        rg = np.random.default_rng(1000 + seed)
        Zg = np.full_like(Zr, np.nan)
        for t in range(T):
            for j in (0, 1):
                w = rg.normal(size=3)                      # always drawn (CRN)
                if np.isfinite(Zr[t, j, 0]):
                    z = ic.h_mono(traj[t], sens[t, j]) + w * np.sqrt(np.diag(R_of(model[j], traj[t], sens[t, j])))
                    z[0] = ic.wrap(z[0])
                    Zg[t, j] = z
        Zc = Zr - np.array([bias[0], bias[1]])[None]      # calibrated real measurements
        cfgs = [("rsu", Zc, [0], None, "anchored"), ("ego", Zc, [1], None, "anchored"),
                ("cen", Zc, [0, 1], None, "anchored"), ("cen_uncal", Zr, [0, 1], None, "anchored"),
                ("cen_gauss", Zg, [0, 1], None, "anchored")]
        cfgs += [("distfull", Zc, [0, 1], None, "anchored")]
        cfgs += [(f"dist{m}", Zc, [0, 1], m, "anchored") for m in (6, 8, 10, 12, 16)]
        cfgs += [(f"abs{m}", Zc, [0, 1], m, "absolute") for m in (8, 12, 16)]
        cfgs += [("dist8_uncal", Zr, [0, 1], 8, "anchored"), ("dist8_gauss", Zg, [0, 1], 8, "anchored")]
        if only is not None:
            cfgs = [c for c in cfgs if c[0] in only]
        for name, Z, use, bits, form in cfgs:
            dist = name.startswith(("dist", "abs"))
            rq = np.random.default_rng(3000 + seed)
            e2 = np.zeros(T)
            ndet = 0
            if not dist:
                mu, P = mu0.copy(), P0.copy()
                Tg = np.zeros((4, 6))                       # dynamic bias gain (cen_uncal)
                pb2 = np.zeros(T)
                for t in range(T):
                    mu = F @ mu; P = F @ P @ F.T + Q
                    Lam, nu = np.linalg.inv(P), np.zeros(4)
                    K = np.zeros((4, 6)); Hs = np.zeros((6, 4))
                    msgs = {}
                    for j in use:
                        m = info(mu, P, Z[t, j], sens[t, j], model[j], kappa)
                        if m is not None:
                            msgs[j] = m; Lam = Lam + m[0]; nu = nu + m[1]; ndet += 1
                    Pp = np.linalg.inv(Lam)
                    for j, m in msgs.items():
                        K[:, 3 * j:3 * j + 3] = Pp @ m[2].T @ m[3]
                        Hs[3 * j:3 * j + 3] = m[2]
                    mu, P = mu + Pp @ nu, Pp
                    Tg = (np.eye(4) - K @ Hs) @ F @ Tg + K
                    pb2[t] = np.sum((S2 @ Tg @ np.r_[bias[0], bias[1]]) ** 2)
                    e2[t] = np.sum((mu[:2] - traj[t, :2]) ** 2)
                if name == "cen_uncal":
                    out.setdefault("prop1_pred_bias2", []).append(float(np.mean(pb2[BURN:])))
            else:
                mus = [mu0.copy(), mu0.copy()]; Ps = [P0.copy(), P0.copy()]
                for t in range(T):
                    for b in (0, 1):
                        mus[b] = F @ mus[b]; Ps[b] = F @ Ps[b] @ F.T + Q
                    c = mus[0].copy() if form == "anchored" else np.zeros(4)
                    infos = []
                    for b in (0, 1):
                        m = info(mus[b], Ps[b], Z[t, b], sens[t, b], model[b], kappa)
                        if m is not None:
                            ndet += 1
                            infos.append((m[0], m[1]))
                        else:
                            infos.append((np.zeros((4, 4)), np.zeros(4)))
                    # B = 2 on one link: one exact averaging round of the quantized messages
                    mus, Ps, _ = ic.dist_fuse(mus, Ps, infos, c, ic.metropolis(ic.full_adj(2)), 1, bits,
                                           0.0, rq, rule="plain", fusion="hcmci")
                    ee = sum(np.sum((mus[b][:2] - traj[t, :2]) ** 2) for b in (0, 1)) / 2
                    e2[t] = ee
            v = ic.rmse(e2, burn=BURN)
            out.setdefault(name, []).append(float(v) if np.isfinite(v) else float("inf"))
            out.setdefault(name + "_mse", []).append(float(np.mean(e2[BURN:])) if np.all(np.isfinite(e2)) else float("inf"))
            out.setdefault(name + "_acc", []).append(ndet)
    rec = {k: float(np.median(v)) for k, v in out.items()}
    det = np.isfinite(tr["Z"][..., 0])
    rec.update(scene=tr["scene"], split=tr["split"], vid=tr["vid"], T=int(T),
               det_rsu=float(det[:, 0].mean()), det_ego=float(det[:, 1].mean()),
               det_both=float(det.all(1).mean()), det_any=float(det.any(1).mean()),
               range_rsu=float(np.median(np.linalg.norm(traj[:, :2] - tr["sens"][:, 0], axis=1))),
               range_ego=float(np.median(np.linalg.norm(traj[:, :2] - tr["sens"][:, 1], axis=1))))
    return rec


def _run(args):
    return run_track(*args)


def _tune(args):
    tr, model, sa, ka = args
    return sa, ka, run_track(tr, model, sa, ka, only=("cen",))["cen"]


def main():
    scs = io.scenes()
    print(len(scs), "scenes", flush=True)
    with Pool(18) as pool:
        parts = pool.map(scene_job, scs)
    resid = [r for split, rs, _ in parts if split == "train" for r in rs]
    model = fit_noise(resid)
    for j in (0, 1):
        m = model[j]
        print(f"agent {j}: sigma_c {m['sigma_c']:.2f} m, sigma_0 {np.degrees(m['sigma_0']):.2f} deg, "
              f"sigma_r {m['sigma_r']:.2f} m, sigma_rr {m['sigma_rr']:.2f} m/s, bias {np.round(m['bias'], 3)} (n={m['n']})")
    train = [t for split, _, ts in parts if split == "train" for t in ts
             if np.isfinite(t["Z"][..., 0]).any(1).mean() >= DET_MIN]
    with Pool(18) as pool:
        tune = pool.map(_tune, [(t, model, sa, ka) for t in train for sa in SIGMA_A_GRID for ka in KAPPA_GRID])
    grid = {}
    for sa, ka, v in tune:
        grid.setdefault((sa, ka), []).append(v)
    grid = {k: float(np.median(v)) for k, v in grid.items()}
    sigma_a = SIGMA_A
    kappa = min((k for k in grid if k[0] == SIGMA_A), key=grid.get)[1]
    print(f"tuned on {len(train)} train tracks: sigma_a={sigma_a}, kappa={kappa}, median RMSE {grid[(sigma_a, kappa)]:.3f}")
    held = [t for split, _, ts in parts if split != "train" for t in ts]
    print(len(held), "held-out tracks", flush=True)
    with Pool(18) as pool:
        recs = pool.map(_run, [(t, model, sigma_a, kappa) for t in held])
    # held-out residuals (model check)
    hr = np.array([r for split, rs, _ in parts if split != "train" for r in rs], float)
    held_resid = {}
    for j in (0, 1):
        A = hr[hr[:, 0] == j]
        held_resid[j] = dict(n=int(len(A)), bias=[float(np.median(A[:, k])) for k in (2, 3, 4)],
                             mad=[float(mad(A[:, k])) for k in (2, 3, 4)])
    OUT.write_text(json.dumps(dict(model={str(k): v for k, v in model.items()},
                                   held_resid={str(k): v for k, v in held_resid.items()},
                                   records=recs, seeds=len(SEEDS), gate=GATE, sigma_a=sigma_a, kappa=kappa,
                                   tune_grid=[[k[0], k[1], v] for k, v in grid.items()],
                                   n_train_tracks=len(train), det_min=DET_MIN)))
    for k in ("rsu", "ego", "cen", "cen_uncal", "cen_gauss", "distfull", "dist6", "dist8", "dist10", "dist12",
              "dist16", "abs8", "abs12", "abs16", "dist8_uncal", "dist8_gauss"):
        v = np.array([r[k] for r in recs])
        w = np.array([r[k] for r in recs if r["det_any"] >= DET_MIN])
        print(f"  {k:12s} all: median {np.median(v):.3f} p90 {np.percentile(v, 90):.3f} | "
              f"det>={DET_MIN}: median {np.median(w):.3f} p90 {np.percentile(w, 90):.3f} (n={len(w)})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
