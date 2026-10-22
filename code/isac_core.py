"""
Core library for the DistTrackingISAC experiments (revision 2026-10).

Model (matches Section II of DistTrackingISAC.tex):
  * target state x = [px, py, vx, vy] with constant-velocity dynamics;
  * each BS b reports a monostatic [AoA, range, range-rate] measurement whose
    covariance follows the CRB-SNR law  R_b = diag(s_ref^2) / SNR_b, with the
    two-way radar budget  SNR_b = g_b * snr_ref * (r_ref / r)^4;
  * centralized EKF (oracle) and a distributed tracker (GBP-CIF) in which every
    BS keeps its own prior and the network computes the sum of the local
    information messages by quantized consensus over a graph with random
    symmetric link failures.

All trackers in one call share the same measurement realisations (common random
numbers), so the reported gaps are not masked by Monte-Carlo noise.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence

D = 4                                    # state dimension
IU = np.triu_indices(D)                  # upper-triangle entries of Lambda (10)
N_LAM = len(IU[0])
N_MSG = N_LAM + D                        # message length per target (14)
# Quantizer range blocks: Lambda_pp, Lambda_pv, Lambda_vv, nu_p, nu_v
_blk = []
for (i, j) in zip(*IU):
    _blk.append(0 if (i < 2 and j < 2) else (2 if (i >= 2 and j >= 2) else 1))
BLOCKS = np.array(_blk + [3, 3, 4, 4])


# ---------------------------------------------------------------- dynamics
def cv_F(dt: float) -> np.ndarray:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def cv_Q(dt: float, sigma_a: float) -> np.ndarray:
    q = sigma_a ** 2
    return q * np.array([[dt**4/4, 0, dt**3/2, 0],
                         [0, dt**4/4, 0, dt**3/2],
                         [dt**3/2, 0, dt**2, 0],
                         [0, dt**3/2, 0, dt**2]])


# ---------------------------------------------------------------- sensing
@dataclass
class BS:
    pos: np.ndarray                      # (2,) metres
    snr_ref: float = 256.0               # echo SNR at r_ref with unit beam gain
    name: str = ""


R_REF = 100.0
SIG_REF = np.array([0.02, 1.0, 0.3])     # [rad, m, m/s] at unit SNR


def h_mono(x, p):
    dx, dy = x[0] - p[0], x[1] - p[1]
    r = np.hypot(dx, dy)
    return np.array([np.arctan2(dy, dx), r, (dx*x[2] + dy*x[3]) / max(r, 1e-6)])


def H_mono(x, p):
    dx, dy = x[0] - p[0], x[1] - p[1]
    r2 = max(dx*dx + dy*dy, 1e-12)
    r = np.sqrt(r2)
    vx, vy = x[2], x[3]
    rr = (dx*vx + dy*vy) / r
    return np.array([[-dy/r2, dx/r2, 0, 0],
                     [dx/r, dy/r, 0, 0],
                     [(vx - rr*dx/r)/r, (vy - rr*dy/r)/r, dx/r, dy/r]])


SNR_MAX = 1e4                            # 40 dB receiver dynamic-range cap


def snr_at(bs: BS, x, gain=1.0, ple=4.0):
    fn = getattr(bs, "snr_fn", None)
    if fn is not None:                   # e.g. ray-traced echo-SNR map
        return min(gain * fn(x), SNR_MAX)
    r = max(np.hypot(x[0]-bs.pos[0], x[1]-bs.pos[1]), 1.0)
    return min(gain * bs.snr_ref * (R_REF / r) ** ple, SNR_MAX)


def R_meas(bs: BS, x, gain=1.0, ple=4.0, snr_floor=1e-3):
    return np.diag(SIG_REF**2) / max(snr_at(bs, x, gain, ple), snr_floor)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def gen_measurements(traj, bss, rng, vis=None, gains=None, ple=4.0):
    """z[t,b] (3,), R[t,b] (3,3) at the true state; NaN rows where invisible."""
    T, B = len(traj), len(bss)
    Z = np.full((T, B, 3), np.nan)
    Rs = np.zeros((T, B, 3, 3))
    for t in range(T):
        for b, bs in enumerate(bss):
            g = 1.0 if gains is None else gains[t, b]
            R = R_meas(bs, traj[t], g, ple)
            Rs[t, b] = R
            w = rng.normal(size=3)        # always drawn: common random numbers
            if vis is not None and not vis[t, b]:
                continue
            z = h_mono(traj[t], bs.pos) + w * np.sqrt(np.diag(R))
            z[0] = wrap(z[0])
            Z[t, b] = z
    return Z, Rs


def local_info(mu, z, bs, R=None):
    """Innovation-form local message (Lambda_b, nu_b) linearised at mu. The
    noise covariance is evaluated at mu (the tracker does not know the truth)."""
    if np.any(np.isnan(z)):
        return np.zeros((4, 4)), np.zeros(4)
    R = R_meas(bs, mu)
    H = H_mono(mu, bs.pos)
    Ri = np.diag(1.0 / np.diag(R))
    innov = z - h_mono(mu, bs.pos)
    innov[0] = wrap(innov[0])
    return H.T @ Ri @ H, H.T @ Ri @ innov


# ---------------------------------------------------------------- messages
def pack(L, v):
    return np.concatenate([L[IU], v])


def unpack(m):
    L = np.zeros((4, 4))
    L[IU] = m[:N_LAM]
    L = L + L.T - np.diag(np.diag(L))
    return L, m[N_LAM:]


CODEC_OVERFLOW = [0]                       # out-of-domain exponent-code events


def quantize_rows(X, bits, rng, range_code="exact"):
    """Subtractive-dithered midtread quantizer, one dynamic range per block and
    per transmitted vector (ranges sent as side information). Returns the
    reconstruction, the per-entry error variance (step^2/12) actually used.
    range_code="exact": the range is the block maximum (x 1.0001), i.e. sent as a float;
    range_code="pow2":  the range is rounded UP to a power of two and sent as a signed
    8-bit exponent (realizable side information; the step can grow by up to 2x)."""
    if bits is None:
        return X.copy(), np.zeros_like(X)
    # 2^b - 1 reconstruction levels {-L..L}*step with L = 2^(b-1) - 1, so a
    # fixed-length code uses exactly b bits; with |x| <= R/1.0001 and dither within
    # half a step, |index| <= round(L/1.0001 + 1/2) = L: the index never overloads.
    # Supported domain of the 8-bit exponent code: block ranges in [2^-126, 2^127].
    # Below 2^-126 (e.g. an all-zero block) the range is floored at 2^-126, which adds
    # a negligible variance floor nu_min; above 2^127 the block is a declared coder
    # failure (NaN reconstruction, counted in CODEC_OVERFLOW), i.e. the run diverges.
    assert bits >= 2
    L = 2 ** (bits - 1) - 1
    Y = np.empty_like(X)
    var = np.empty_like(X)
    for k in range(BLOCKS.max() + 1):
        idx = BLOCKS == k
        R = np.max(np.abs(X[:, idx]), axis=1, keepdims=True) * 1.0001 + 1e-300
        if range_code == "pow2":
            ex = np.ceil(np.log2(R))
            over = ~(ex <= 127)                     # outside the code (incl. diverged inf/NaN)
            if np.any(over):
                CODEC_OVERFLOW[0] += int(over.sum())
            R = np.where(over, np.nan, 2.0 ** np.maximum(ex, -126))   # declared failure: NaN
        step = R / L
        d = (rng.uniform(size=X[:, idx].shape) - 0.5) * step
        Y[:, idx] = np.round((X[:, idx] + d) / step) * step - d
        var[:, idx] = np.broadcast_to(step**2 / 12.0, X[:, idx].shape)
    return Y, var


# ---------------------------------------------------------------- graphs
def metropolis(adj):
    B = adj.shape[0]
    deg = adj.sum(1)
    W = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i != j and adj[i, j]:
                W[i, j] = 1.0 / (1 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W


def full_adj(B):
    return np.ones((B, B)) - np.eye(B)


def ring_adj(B):
    A = np.zeros((B, B))
    for i in range(B):
        A[i, (i+1) % B] = A[(i+1) % B, i] = 1
    return A


def line_adj(B):
    A = np.zeros((B, B))
    for i in range(B-1):
        A[i, i+1] = A[i+1, i] = 1
    return A


def sample_W(W, p_fail, rng):
    """Symmetric link failures: a failed edge's weight returns to both ends."""
    if p_fail <= 0:
        return W
    Wi = W.copy()
    B = W.shape[0]
    for i in range(B):
        for j in range(i+1, B):
            if W[i, j] > 0 and rng.random() < p_fail:
                w = W[i, j]
                Wi[i, j] = Wi[j, i] = 0.0
                Wi[i, i] += w
                Wi[j, j] += w
    return Wi


def link_moments(W, p_fail):
    """Exact E[C^2] and E[(I-C)^2] for independent symmetric edge failures (the model of
    sample_W): C = I - sum_e s_e w_e b_e b_e^T, s_e ~ Bernoulli(1-p), (b_e b_e^T)^2 = 2 b_e b_e^T.
    With L = sum_e w_e b_e b_e^T, L2 = sum_e w_e^2 b_e b_e^T, q = 1-p:
    E[C^2] = (I - qL)^2 + 2pq L2,  E[(I-C)^2] = q^2 L^2 + 2pq L2."""
    B = W.shape[0]
    L = np.zeros((B, B)); L2 = np.zeros((B, B))
    for i in range(B):
        for j in range(i + 1, B):
            if W[i, j] > 0:
                b = np.zeros(B); b[i], b[j] = 1.0, -1.0
                L += W[i, j] * np.outer(b, b); L2 += W[i, j] ** 2 * np.outer(b, b)
    q = 1.0 - p_fail
    I = np.eye(B)
    return (I - q * L) @ (I - q * L) + 2 * p_fail * q * L2, q * q * L @ L + 2 * p_fail * q * L2


def lambda_bar(W, p_fail, **_):
    """||E[C(i)^2] - J_B||: per-round mean-square contraction on 1-perp (exact)."""
    EC2, _ = link_moments(W, p_fail)
    B = W.shape[0]
    return float(np.linalg.norm(EC2 - np.ones((B, B)) / B, 2))


def consensus(X0, W, I, bits, p_fail, rng, rule="compensating", range_code="exact"):
    """Every transmission is quantized (Xq = X + E).
    rule="plain":        X(i+1) = W(i) Xq(i)              (node keeps its own quantized value)
    rule="compensating": X(i+1) = X(i) + (W(i) - I) Xq(i) (average preserved exactly;
                                                          Frasca et al. 2009, Carli et al. 2010)
    Returns X(I) and the per-round quantization variances V(i) for Theorem 2."""
    X = X0.copy()
    V = []
    for _ in range(I):
        Xq, var = quantize_rows(X, bits, rng, range_code)
        V.append(var.sum())
        Wi = sample_W(W, p_fail, rng)
        X = Wi @ Xq if rule == "plain" else X + Wi @ Xq - Xq
    return X, np.array(V)


def thm2_bound(X0, V, lam, B, rule="plain", rho=1.0):
    """Right-hand side of Theorem 2 for E||X(I) - 1 s^T||_F^2.
    plain:        lam^I E0 + sum_i (1/B + (1 - 1/B) lam^(I-i)) V(i)
    compensating: lam^I E0 + sum_i rho lam^(I-1-i) V(i),  rho = max_b E[(I - C)^2]_bb"""
    s = X0.mean(0)
    I = len(V)
    dev = np.sum((X0 - s) ** 2)
    if rule == "plain":
        acc = sum((1.0 / B + (1.0 - 1.0 / B) * lam ** (I - i)) * V[i] for i in range(I))
    else:
        acc = sum(rho * lam ** (I - 1 - i) * V[i] for i in range(I))
    return lam ** I * dev + acc


def rho_bar(W, p_fail, **_):
    """kappa_bar = max_b E[(I - C(i))^2]_bb: per-round gain of the compensating rule's noise (exact)."""
    _, EIC2 = link_moments(W, p_fail)
    return float(np.max(np.diag(EIC2)))


def dist_fuse(mus, Ps, infos, c, W, I, bits, p_fail, rng, rule="plain", fusion="hcmci", range_code="exact"):
    """One distributed update for B nodes from their (predicted) beliefs and local
    measurement information infos[b] = (Lambda_b, nu_b) (zeros if no measurement).
    Messages are anchored at c (c = 0 gives absolute messages). A node whose fused
    information is not positive definite keeps its prediction (declared safeguard).
    Returns new (mus, Ps, number of such fallbacks)."""
    B = len(mus)
    X0 = np.zeros((B, N_MSG))
    for b in range(B):
        Lb, nb = infos[b]
        Lpr = np.linalg.inv(Ps[b])
        if fusion == "hcmci":
            X0[b] = B * pack(Lpr / B + Lb, (Lpr / B + Lb) @ (mus[b] - c) + nb)
        elif fusion == "hcmci_avg":
            X0[b] = pack(Lpr + Lb, (Lpr + Lb) @ (mus[b] - c) + nb)
        else:
            X0[b] = B * pack(Lb, nb + Lb @ (mus[b] - c))
    X, _ = consensus(X0, W, I, bits, p_fail, rng, rule, range_code)
    out_mu, out_P, nfail = [], [], 0
    for b in range(B):
        Lt, vt = unpack(X[b])
        Lpr = np.zeros((4, 4)) if fusion.startswith("hcmci") else np.linalg.inv(Ps[b])
        Lp = Lpr + Lt
        try:
            np.linalg.cholesky(0.5 * (Lp + Lp.T))
        except np.linalg.LinAlgError:
            out_mu.append(mus[b]); out_P.append(Ps[b]); nfail += 1
            continue
        Pp = np.linalg.inv(Lp)
        out_mu.append(c + Pp @ (Lpr @ (mus[b] - c) + vt))
        out_P.append(Pp)
    return out_mu, out_P, nfail


# ---------------------------------------------------------------- trackers
@dataclass
class FusionCfg:
    name: str
    adj: Optional[np.ndarray] = None     # None => centralized
    I: int = 1
    bits: Optional[int] = None
    p_fail: float = 0.0
    form: str = "anchored"               # or "absolute" (anchor at the origin)
    rule: str = "compensating"           # consensus update: "compensating" or "plain"
    fusion: str = "hcmci"                # "cm": consensus on new information, own prior kept
                                         # "hcmci": prior information / B averaged as well
                                         # (Battistelli et al. 2015, merged single consensus;
                                         # weight B: centralized limit, NOT consistency-preserving)
                                         # "hcmci_avg": average prior AND average measurement
                                         # information (weight 1; consistency-preserving variant)
    range_code: str = "exact"            # "exact" (float ranges) or "pow2" (8-bit exponents)
    anchor_lag: int = 0                  # 0: leader's current prediction; 1: pipelined anchor
                                         # c_t = F^2 mu_leader(t-2), disseminated during update t-1


def run_trackers(traj, bss, dt, cfgs: Sequence[FusionCfg], seed=0,
                 sigma_a=1.0, vis=None, P0=None, x0_sd=(2, 2, 1, 1),
                 diag=False, origin=None, keep_mu=False, init_spread=None, spd_counts=None):
    """Run every configuration on the same measurement realisation.
    `origin` (2,) translates the whole scene (to test frame dependence).
    `init_spread` (4,): if given, every BS of a distributed tracker starts from its
    own prior mean mu0 + N(0, diag(init_spread^2)) (non-common priors).
    Returns dict name -> (T, 2) position errors averaged over BSs (squared),
    plus optional diagnostics. If `spd_counts` is a dict, it receives, per distributed
    configuration, the number of node updates rejected by the SPD safeguard."""
    rng_z = np.random.default_rng(1000 + seed)
    traj = traj.copy()
    bss_t = bss
    if origin is not None:
        traj[:, :2] += origin
        bss_t = [BS(b.pos + origin, b.snr_ref, b.name) for b in bss]
    Z, Rs = gen_measurements(traj, bss_t, rng_z, vis)
    T, B = len(traj), len(bss_t)
    F, Q = cv_F(dt), cv_Q(dt, sigma_a)
    P0 = np.diag([25.0, 25.0, 4.0, 4.0]) if P0 is None else P0
    rng0 = np.random.default_rng(2000 + seed)
    mu0 = traj[0] + rng0.normal(size=4) * np.array(x0_sd)
    out, diags = {}, {}
    for cfg in cfgs:
        rng_q = np.random.default_rng(3000 + seed)
        dist = cfg.adj is not None
        nB = B if dist else 1
        mus = [mu0.copy() for _ in range(nB)]
        if dist and init_spread is not None:
            rng_i = np.random.default_rng(4000 + seed)
            mus = [mu0 + rng_i.normal(size=4) * np.asarray(init_spread) for _ in range(nB)]
        Ps = [P0.copy() for _ in range(nB)]
        if dist:
            W = metropolis(cfg.adj)
        err2 = np.zeros((T, nB))
        nees = np.zeros((T, nB))
        mu_hist = np.zeros((T, nB, 4))
        dg = {"act": [], "bnd": [], "dev0": [], "V": [], "omega": [], "spdfail": []}
        lead_hist = []                               # leader posterior means (pipelined anchor)
        lam = lambda_bar(W, cfg.p_fail) if (dist and diag) else None
        rho = rho_bar(W, cfg.p_fail) if (dist and diag and cfg.rule != "plain") else 1.0
        for t in range(T):
            for k in range(nB):
                mus[k] = F @ mus[k]
                Ps[k] = F @ Ps[k] @ F.T + Q
            if not dist:
                Lam = np.zeros((4, 4)); nu = np.zeros(4)
                for b in range(B):
                    Lb, nb = local_info(mus[0], Z[t, b], bss_t[b], Rs[t, b])
                    Lam += Lb; nu += nb
                Lp = np.linalg.inv(Ps[0]) + Lam
                Ps[0] = np.linalg.inv(Lp)
                mus[0] = mus[0] + Ps[0] @ nu
            else:
                # common anchor: leader's predicted mean (2 scalars flooded per
                # step) or the coordinate origin for the "absolute" form
                if cfg.form == "absolute":
                    c = np.zeros(4)
                elif cfg.anchor_lag > 0 and len(lead_hist) > cfg.anchor_lag:
                    # pipelined anchor: leader's posterior from lag+1 updates ago, predicted
                    # forward; disseminated during the previous update (any common c is exact)
                    c = np.linalg.matrix_power(F, cfg.anchor_lag + 1) @ lead_hist[-1 - cfg.anchor_lag]
                else:
                    c = mus[0].copy()           # (pipelined mode: startup, first lag+1 updates)
                if cfg.anchor_lag > 0 and cfg.form != "absolute":
                    c = c.astype(np.float32).astype(np.float64)   # sent as four binary32 values
                X0 = np.zeros((B, N_MSG))
                for b in range(B):
                    Lb, nb = local_info(mus[b], Z[t, b], bss_t[b], Rs[t, b])
                    if cfg.fusion == "hcmci":
                        Lpr = np.linalg.inv(Ps[b])
                        X0[b] = B * pack(Lpr / B + Lb, (Lpr / B + Lb) @ (mus[b] - c) + nb)
                    elif cfg.fusion == "hcmci_avg":     # weight 1: averages of prior and new info
                        Lpr = np.linalg.inv(Ps[b])
                        X0[b] = pack(Lpr + Lb, (Lpr + Lb) @ (mus[b] - c) + nb)
                    else:
                        X0[b] = B * pack(Lb, nb + Lb @ (mus[b] - c))
                X, V = consensus(X0, W, cfg.I, cfg.bits, cfg.p_fail, rng_q, cfg.rule, cfg.range_code)
                if diag:
                    dg["act"].append(np.sum((X - X0.mean(0)) ** 2))
                    dg["bnd"].append(thm2_bound(X0, V, lam, B, cfg.rule, rho))
                    dg["dev0"].append(np.sum((X0 - X0.mean(0)) ** 2))
                    dg["V"].append(np.mean(V) if len(V) else 0.0)
                if diag:
                    # realised Theorem-3 margin: max_b ||P*_b|| ||Xi~_b - Xi*||. For HCMCI the
                    # exact network value Xi* already contains the (averaged) prior, so P* = Xi*^-1;
                    # only CM adds BS b's own prior.
                    Lstar, _ = unpack(X0.mean(0))
                    om = 0.0
                    for b in range(B):
                        Lt_b, _ = unpack(X[b])
                        Pst = (np.linalg.inv(Lstar) if cfg.fusion.startswith("hcmci")
                               else np.linalg.inv(np.linalg.inv(Ps[b]) + Lstar))
                        om = max(om, np.linalg.norm(Pst, 2) * np.linalg.norm(Lt_b - Lstar, 2))
                    dg["omega"].append(om)
                nfail = 0
                for b in range(B):
                    Lt, vt = unpack(X[b])
                    Lprior = np.linalg.inv(Ps[b])
                    if cfg.fusion.startswith("hcmci"):  # fused prior + measurement information
                        Lprior = np.zeros((4, 4))
                    Lp = Lprior + Lt
                    try:
                        np.linalg.cholesky(0.5 * (Lp + Lp.T))   # fused information must be SPD
                    except np.linalg.LinAlgError:
                        nfail += 1                     # declared safeguard: keep the prediction
                        continue
                    Pp = np.linalg.inv(Lp)
                    mus[b] = c + Pp @ (Lprior @ (mus[b] - c) + vt)
                    Ps[b] = Pp
                lead_hist.append(mus[0].copy())
                if spd_counts is not None:
                    spd_counts[cfg.name] = spd_counts.get(cfg.name, 0) + nfail
                if diag:
                    dg["spdfail"].append(nfail)
            for k in range(nB):
                mu_hist[t, k] = mus[k]
                e = mus[k][:2] - traj[t, :2]
                err2[t, k] = e @ e if np.all(np.isfinite(e)) else np.inf
                if diag:
                    try:
                        nees[t, k] = e @ np.linalg.solve(Ps[k][:2, :2], e)
                    except np.linalg.LinAlgError:
                        nees[t, k] = np.inf
        out[cfg.name] = err2.mean(1)
        if diag:
            out[cfg.name + "_nees"] = nees.mean(1)          # position NEES (2 dof if consistent)
            out[cfg.name + "_cov"] = (nees <= 5.991).mean(1)  # 95% position-ellipse coverage
        if keep_mu:
            out[cfg.name + "_mu"] = mu_hist
        if diag and dist:
            diags[cfg.name] = {k: np.array(v) for k, v in dg.items()}
    return (out, diags) if diag else out


def rmse(err2, burn=20):
    e = err2[burn:]
    return float(np.sqrt(np.mean(e))) if np.all(np.isfinite(e)) else float("inf")


# ---------------------------------------------------------------- bounds
def pcrb(traj, bss, dt, sigma_a=1.0, P0=None, vis=None):
    F, Q = cv_F(dt), cv_Q(dt, sigma_a)
    P0 = np.diag([25.0, 25.0, 4.0, 4.0]) if P0 is None else P0
    J = np.linalg.inv(P0)
    out = []
    for t in range(len(traj)):
        J = np.linalg.inv(F @ np.linalg.inv(J) @ F.T + Q)
        for b, bs in enumerate(bss):
            if vis is not None and not vis[t, b]:
                continue
            H = H_mono(traj[t], bs.pos)
            J = J + H.T @ np.linalg.inv(R_meas(bs, traj[t])) @ H
        out.append(np.linalg.inv(J))
    return np.array(out)


def window_gramian(traj, t, K, bs_list, dt, vis_rows=None):
    """Observability Gramian of the state at t from a window of K snapshots:
    G = sum_tau Phi^T H^T R^-1 H Phi, Phi = F^(tau - t). `bs_list` is a list
    of BSs or a callable tau -> list (moving agents); `vis_rows[tau, j]` masks
    snapshots in which agent j does not observe the target."""
    F = cv_F(dt)
    G = np.zeros((4, 4))
    for k in range(K):
        tau = t - k
        if tau < 0:
            break
        Phi = np.linalg.matrix_power(np.linalg.inv(F), k)
        bl = bs_list(tau) if callable(bs_list) else bs_list
        for j, bs in enumerate(bl):
            if vis_rows is not None and not vis_rows[tau, j]:
                continue
            H = H_mono(traj[tau], bs.pos)
            G += Phi.T @ H.T @ np.linalg.inv(R_meas(bs, traj[tau])) @ H @ Phi
    return G


def coop_bound(A, Bm, tol=1e-9):
    """Lemma 1: lambda_min(A+B) >= beta*lam_plus/(2||B|| + lam_plus) where
    lam_plus is A's smallest positive eigenvalue and beta = lambda_min of B
    compressed onto ker(A). Falls back to Weyl when A is nonsingular."""
    ev, U = np.linalg.eigh(A)
    scale = max(ev.max(), 1e-300)
    null = ev < tol * scale
    if not null.any():
        return float(ev.min() + max(np.linalg.eigvalsh(Bm).min(), 0.0))
    Un = U[:, null]
    beta = float(np.linalg.eigvalsh(Un.T @ Bm @ Un).min())
    lam_plus = float(ev[~null].min())
    nB = float(np.linalg.eigvalsh(Bm).max())
    return max(beta, 0.0) * lam_plus / (2 * nB + lam_plus)


# ---------------------------------------------------------------- scenes
def ring_bss(center, R, B, snr_ref=256.0):
    a = np.linspace(0, 2*np.pi, B, endpoint=False)
    return [BS(center + R*np.array([np.cos(t), np.sin(t)]), snr_ref, f"BS{i}")
            for i, t in enumerate(a)]


def synth_trajectory(T, dt, seed=1, v0=(15.0, 2.0)):
    x = np.array([0.0, 0.0, *v0])
    F = cv_F(dt)
    out = [x.copy()]
    for t in range(1, T):
        th = 0.05 * np.sin(2*np.pi*t/80) * dt
        c, s = np.cos(th), np.sin(th)
        x[2:] = np.array([[c, -s], [s, c]]) @ x[2:]
        x = F @ x
        out.append(x.copy())
    return np.array(out)


def provenance(driver_file):
    """Run metadata stored with every result file: source hashes, versions, time."""
    import hashlib, platform, sys, datetime, os
    import scipy
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    here = os.path.dirname(os.path.abspath(__file__))
    return dict(driver=os.path.basename(driver_file), driver_sha256=h(driver_file),
                isac_core_sha256=h(os.path.join(here, "isac_core.py")),
                python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                platform=platform.platform(), omp_threads=os.environ.get("OMP_NUM_THREADS", "default"),
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                spd_safeguard=True)
