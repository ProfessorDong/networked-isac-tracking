"""
Experiment J: worst-case MSE under ellipsoidal calibration uncertainty
(Proposition 1).

Each BS b has a constant, unknown bias phi_b in its [AoA, range, range-rate]
report. phi = [phi_1;...;phi_B] lies in U = {phi : phi^T Omega phi <= 1} with
Omega = rho^{-2} blkdiag(S_b^{-2}), S_b = diag of the measurement standard
deviations at the scene centre, so rho is the bias radius in units of the
nominal noise level.

Trackers: naive EKF (ignores phi), joint EKF (state augmented with phi, prior
covariance Omega^{-1}). Predictions:
  dynamic (Prop. 1):   MSE_wc = MSE* + rho^2 lambda_max( Omega~^{-1/2} Gbar Omega~^{-1/2} ),
                       Gbar = (1/T) sum_t G_t^T S^T S G_t,  G_t = (I-K_t H_t) F G_{t-1} + K_t
  static snapshot:     same with G_t replaced by the LS gain (old Corollary).
The naive tracker is driven with the worst-case direction from the dynamic
prediction and with random directions.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from isac_core import (synth_trajectory, ring_bss, cv_F, cv_Q, h_mono, H_mono, R_meas,
                       wrap, SIG_REF, snr_at)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_J.json"
T, DT, B = 150, 0.1, 4
TRAJ = synth_trajectory(T, DT)
BSS = ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
BURN = 20
RHOS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
F, Q = cv_F(DT), cv_Q(DT, 1.0)
P0 = np.diag([25.0, 25.0, 4.0, 4.0])
cen = TRAJ[:, :2].mean(0)
S_nom = np.concatenate([SIG_REF / np.sqrt(snr_at(b, np.r_[cen, 0, 0])) for b in BSS])
OM_INV1 = np.diag(S_nom ** 2)            # Omega^{-1} for rho = 1


def gains_along_truth():
    """Kalman gains of the centralised filter linearised on the true path."""
    P = P0.copy()
    G = np.zeros((4, 3 * B))
    Gs, Ks = [], []
    for t in range(T):
        P = F @ P @ F.T + Q
        H = np.vstack([H_mono(TRAJ[t], b.pos) for b in BSS])
        R = np.zeros((3 * B, 3 * B))
        for i, b in enumerate(BSS):
            R[3*i:3*i+3, 3*i:3*i+3] = R_meas(b, TRAJ[t])
        Ri = np.linalg.inv(R)
        Pn = np.linalg.inv(np.linalg.inv(P) + H.T @ Ri @ H)
        K = Pn @ H.T @ Ri
        G = (np.eye(4) - K @ H) @ F @ G + K
        Kls = np.linalg.solve(H.T @ Ri @ H, H.T @ Ri)
        Gs.append(G.copy()); Ks.append(Kls)
        P = Pn
    return Gs, Ks


def wc_direction(Gs):
    S = np.zeros((2, 4)); S[0, 0] = S[1, 1] = 1
    Gbar = sum(G.T @ S.T @ S @ G for G in Gs[BURN:]) / (T - BURN)
    O = np.sqrt(OM_INV1)
    lam, V = np.linalg.eigh(O @ Gbar @ O)
    return float(lam[-1]), O @ V[:, -1]


def run(args):
    kind, rho, seed, phi_dir = args
    rng = np.random.default_rng(seed)
    phi = rho * np.asarray(phi_dir)
    aug = kind == "joint"
    d = 4 + 3 * B if aug else 4
    Fa = np.eye(d); Fa[:4, :4] = F
    Qa = np.zeros((d, d)); Qa[:4, :4] = Q
    mu = np.zeros(d); mu[:4] = TRAJ[0] + rng.normal(size=4) * np.array([2, 2, 1, 1])
    P = np.zeros((d, d)); P[:4, :4] = P0
    if aug:
        P[4:, 4:] = max(rho, 0.05) ** 2 * OM_INV1
    err2 = []
    for t in range(T):
        mu = Fa @ mu; P = Fa @ P @ Fa.T + Qa
        Lam = np.linalg.inv(P); nu = np.zeros(d)
        for i, b in enumerate(BSS):
            R = R_meas(b, TRAJ[t])
            z = h_mono(TRAJ[t], b.pos) + phi[3*i:3*i+3] + rng.normal(size=3) * np.sqrt(np.diag(R))
            Hx = H_mono(mu[:4], b.pos)
            Ha = np.zeros((3, d)); Ha[:, :4] = Hx
            pred = h_mono(mu[:4], b.pos)
            if aug:
                Ha[:, 4+3*i:7+3*i] = np.eye(3)
                pred = pred + mu[4+3*i:7+3*i]
            inn = z - pred; inn[0] = wrap(inn[0])
            Ri = np.linalg.inv(R_meas(b, mu[:4]))      # noise level at the prediction
            Lam += Ha.T @ Ri @ Ha; nu += Ha.T @ Ri @ inn
        P = np.linalg.inv(Lam); mu = mu + P @ nu
        e = mu[:2] - TRAJ[t, :2]
        err2.append(e @ e)
    return float(np.mean(err2[BURN:]))


def main():
    Gs, Ks = gains_along_truth()
    lam_dyn, wdir = wc_direction(Gs)
    lam_sta, _ = wc_direction(Ks)
    rng = np.random.default_rng(7)
    rand_dirs = []
    for _ in range(20):
        v = rng.normal(size=3 * B)
        v = np.sqrt(OM_INV1) @ (v / np.linalg.norm(v))       # on the ellipsoid boundary
        rand_dirs.append(v)
    jobs = []
    for rho in RHOS:
        for s in range(10):
            jobs.append(("naive_wc", rho, s, wdir))
            jobs.append(("joint_wc", rho, s, wdir))
        for s, v in enumerate(rand_dirs):
            jobs.append(("naive_rand", rho, 100 + s, v))
    jobs = [("joint" if k.startswith("joint") else "naive", r, s, v) for (k, r, s, v) in jobs]
    labels = []
    for rho in RHOS:
        labels += [("naive_wc", rho), ("joint_wc", rho)] * 10 + [("naive_rand", rho)] * 20
    with Pool(18) as pool:
        mses = pool.map(run, jobs)
    rows = {}
    for (lab, rho), m in zip(labels, mses):
        rows.setdefault(f"{lab}_{rho}", []).append(m)
    mse0 = np.mean(rows["naive_wc_0.0"])
    out = dict(rhos=RHOS, lam_dyn=lam_dyn, lam_static=lam_sta, mse0=mse0,
               rows={k: [float(np.mean(v)), float(np.std(v))] for k, v in rows.items()})
    for rho in RHOS:
        print(f"rho={rho:4.1f} naive_wc {np.mean(rows[f'naive_wc_{rho}']):.4f} "
              f"pred_dyn {mse0 + rho**2*lam_dyn:.4f} pred_static {mse0 + rho**2*lam_sta:.4f} "
              f"rand {np.mean(rows[f'naive_rand_{rho}']):.4f} joint {np.mean(rows[f'joint_wc_{rho}']):.4f}")
    OUT.write_text(json.dumps(out))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
