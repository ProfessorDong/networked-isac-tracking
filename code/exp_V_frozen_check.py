"""
Experiment V: direct Monte Carlo check of Theorems 2-3 and Corollary 3 on FROZEN messages.

Theorem 2 bounds an expectation over the consensus randomness of one update, so we freeze
the input message matrix X(0) and repeat only that update's dithers and link failures.
Message sets: B = 6 BSs on the 300-m ring of Exp B, unequal local priors (each BS's
prior mean is the true state plus N(0, P)), measurements drawn once, anchored HCMCI
messages (anchor = BS 0's prior mean). Two time steps of the trajectory are used.

For each (m, I, p, rule) and N independent draws we report
  * Theorem 2: mean ||X(I) - 1 s^T||_F^2 (95% CI) against the bound evaluated with the
    Monte Carlo means of V(i);
  * Theorem 3: max |identity residual|; realised omega_bar = max_draws max_b omega_b;
    and, if omega_bar < 1, the sandwich (iii) with D_b = T(u) E[z z^T] T(u)^T built from
    the FULL second moment of z_b = [vech(Delta_b); delta_b] (bias and correlation kept);
  * Corollary 3: sum_b E||mu_b - mu*||^2 <= ||P*||^2 (1 + 2||u||^2) / (1 - omega_bar)^2
    * E||X(I) - 1 s^T||_F^2;
  * the adaptive-range bound V(i) <= alpha_m ||X(i)||_F^2, alpha_m = r a^2 / (12 L_m^2).
"""
import json
import numpy as np
from pathlib import Path
import isac_core as ic

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_V.json"
B, DT, T = 6, 0.1, 200
TRAJ = ic.synth_trajectory(T, DT)
BSS = ic.ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
W = ic.metropolis(ic.ring_adj(B))
N = 2000
TIMES = (60, 150)
CONF = [(m, I, p, r) for m in (4, 8) for I in (1, 3, 8, 21) for p in (0.0, 0.2)
        for r in ("compensating", "plain")]
IU = ic.IU


def frozen_messages(t, seed=11):
    """Anchored HCMCI messages at time t with unequal priors."""
    rng = np.random.default_rng(seed + t)
    F, Q = ic.cv_F(DT), ic.cv_Q(DT, 1.0)
    # reference covariance: run a centralized EKF on the trajectory up to t
    Z, _ = ic.gen_measurements(TRAJ, BSS, np.random.default_rng(1000 + seed), None)
    mu, P = TRAJ[0].copy(), np.diag([25.0, 25.0, 4.0, 4.0])
    for k in range(t):
        mu, P = F @ mu, F @ P @ F.T + Q
        Lam, nu = np.linalg.inv(P), np.zeros(4)
        for b in range(B):
            Lb, nb = ic.local_info(mu, Z[k, b], BSS[b])
            Lam, nu = Lam + Lb, nu + nb
        P = np.linalg.inv(Lam); mu = mu + P @ nu
    mu, P = F @ mu, F @ P @ F.T + Q                      # prediction at t
    mus = [TRAJ[t] + np.linalg.cholesky(P) @ rng.normal(size=4) for _ in range(B)]
    c = mus[0].copy()
    X0 = np.zeros((B, ic.N_MSG))
    for b in range(B):
        Lb, nb = ic.local_info(mus[b], Z[t, b], BSS[b])
        Lpr = np.linalg.inv(P)
        X0[b] = B * ic.pack(Lpr / B + Lb, (Lpr / B + Lb) @ (mus[b] - c) + nb)
    return X0, c


def Gmat(u):
    """G(u) vech(Delta) = Delta u for symmetric Delta (upper-triangle packing of isac_core)."""
    G = np.zeros((4, len(IU[0])))
    for e, (i, j) in enumerate(zip(*IU)):
        G[i, e] += u[j]
        if i != j:
            G[j, e] += u[i]
    return G


def check(X0, m, I, p, rule, seed):
    rng = np.random.default_rng(seed)
    s = X0.mean(0)
    Xi_s, xi_s = ic.unpack(s)
    Ps = np.linalg.inv(Xi_s); u = Ps @ xi_s; nPs = np.linalg.norm(Ps, 2)
    E0 = np.sum((X0 - s) ** 2)
    lam, kap = ic.lambda_bar(W, p), ic.rho_bar(W, p)
    errs, Vs, dev, om, resid, Z, vratio = [], [], [], [], 0.0, [], 0.0
    L_m = 2 ** (m - 1) - 1
    alpha = 4 * 1.0001 ** 2 / (12 * L_m ** 2)
    for _ in range(N):
        X = X0.copy(); V = []
        for i in range(I):
            Xq, var = ic.quantize_rows(X, m, rng)
            V.append(var.sum())
            vratio = max(vratio, var.sum() / max(np.sum(X ** 2), 1e-300) / alpha)
            Wi = ic.sample_W(W, p, rng)
            X = Wi @ Xq if rule == "plain" else X + Wi @ Xq - Xq
        errs.append(np.sum((X - s) ** 2)); Vs.append(V)
        d_b, o_b, z_b = [], [], []
        for b in range(B):
            Xi_b, xi_b = ic.unpack(X[b])
            D, dl = Xi_b - Xi_s, xi_b - xi_s
            o_b.append(nPs * np.linalg.norm(D, 2))
            mu_b = np.linalg.solve(Xi_b, xi_b)                       # mu_b - c
            rhs = np.linalg.solve(np.eye(4) + Ps @ D, Ps @ (dl - D @ u))
            resid = max(resid, np.max(np.abs((mu_b - u) - rhs)) / max(np.linalg.norm(u), 1e-12))
            d_b.append(np.sum((mu_b - u) ** 2))
            z_b.append(np.r_[X[b][:len(IU[0])] - s[:len(IU[0])], dl])
        dev.append(d_b); om.append(o_b); Z.append(z_b)
    errs, Vs, dev, om, Z = map(np.array, (errs, Vs, dev, om, Z))
    # the bound is linear in V(i): evaluate it per draw, so error and bound are PAIRED and
    # the mean of the per-draw bounds is the bound with E[V(i)] estimated by the same draws
    if rule == "plain":
        coef = np.array([1 / B + (1 - 1 / B) * lam ** (I - i) for i in range(I)])
    else:
        coef = np.array([kap * lam ** (I - 1 - i) for i in range(I)])
    bnd_k = lam ** I * E0 + Vs @ coef
    bnd = bnd_k.mean()
    dk = errs - bnd_k                                         # paired error - bound
    d_hi = dk.mean() + 1.96 * dk.std(ddof=1) / np.sqrt(N)     # upper 95% limit of E[err - bound]
    se = errs.std(ddof=1) / np.sqrt(N)
    res = dict(m=m, I=I, p=p, rule=rule, E_err=float(errs.mean()), ci=float(1.96 * se),
               bound=float(bnd), ratio=float(errs.mean() / bnd), ratio_hi=float(1 + d_hi / bnd),
               d_hi=float(d_hi),
               resid=float(resid), omega_bar=float(om.max()),   # observed (sample) maximum
               alpha_ratio_max=float(vratio))
    wbar = om.max()
    if wbar < 1:
        T_ = np.hstack([-Gmat(u), np.eye(4)])
        lo, hi = [], []
        for b in range(B):
            Mz = Z[:, b, :].T @ Z[:, b, :] / N                       # full second moment
            Db = T_ @ Mz @ T_.T
            tr = np.trace(Ps @ Db @ Ps)
            lo.append(dev[:, b].mean() / (tr / (1 + wbar) ** 2))
            hi.append(dev[:, b].mean() / (tr / (1 - wbar) ** 2))
        cor = nPs ** 2 * (1 + 2 * u @ u) / (1 - wbar) ** 2 * errs.mean()
        res.update(sand_lo_min=float(min(lo)), sand_hi_max=float(max(hi)),
                   cor_ratio=float(dev.sum(1).mean() / cor))
    return res


def main():
    out = []
    for t in TIMES:
        X0, _ = frozen_messages(t)
        Xi_s, _ = ic.unpack(X0.mean(0))
        cond = float(np.linalg.cond(Xi_s))
        for k, (m, I, p, r) in enumerate(CONF):
            res = check(X0, m, I, p, r, seed=10_000 * t + k)
            res.update(t=t, cond=cond)
            out.append(res)
            print(f"t={t} m={m} I={I:2d} p={p} {r[:4]}: ratio {res['ratio']:.3f} (hi {res['ratio_hi']:.3f}) "
                  f"resid {res['resid']:.1e} omega {res['omega_bar']:.2e} alpha {res['alpha_ratio_max']:.3f} "
                  f"sand [{res.get('sand_lo_min', np.nan):.3f}, {res.get('sand_hi_max', np.nan):.3f}] "
                  f"cor {res.get('cor_ratio', np.nan):.3f}", flush=True)
    OUT.write_text(json.dumps(dict(N=N, B=B, times=TIMES, rows=out, provenance=ic.provenance(__file__))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
