"""
Belief-state model-predictive ISAC beamforming (Section VI of the manuscript).

Decision variable: transmit beamformers W[tau, b, k] in C^M for the H slots of
the horizon, BS b, user k of BS b. Per-slot per-BS power  sum_k ||W||^2 <= P.

Objective (per-slot average over the horizon):
    f(W) = eta * U_com(W) / H + (1 - eta) * U_sen(W) / H
    U_com = sum_tau sum_users alpha log2(1 + SINR)
    U_sen = sum_tau sum_q log det J_q[tau]
    J_q[tau] = phi(J_q[tau-1]) + sum_b c_bq[tau] g_bq(W[tau]) E_bq[tau]
    phi(J) = (F J^{-1} F^T + Q)^{-1}            (Bayesian information prediction)
    g_bq(W) = (M/P) sum_k |a_bq^H W[tau,b,k]|^2 (normalised transmit beam gain)
J_q[-1] is the current posterior information of target q. Because J_q[tau] is
bilinear in W through g_bq, the per-slot structure is the one of [ParetoISAC]
with the predictive prior phi(J_q[tau-1]) in place of a fixed J_prior.

Algorithm: proximal SCA with WMMSE auxiliaries, a first-order (w-space)
linearisation of U_sen whose gradient is computed by an adjoint (backward)
recursion through phi, and an adaptive proximal weight rho certified by the
minorisation test f(W+) >= g(W+; W, rho) at every iteration (Theorem 4).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from isac_core import cv_F, cv_Q, H_mono, SIG_REF, R_REF

SINV = np.diag(1.0 / SIG_REF**2)


def steer(M, phi):
    n = np.arange(M) - (M - 1) / 2
    return np.exp(1j * np.pi * n * np.sin(phi)) / np.sqrt(M)


@dataclass
class Scene:
    bs_pos: np.ndarray          # (B, 2)
    bs_az: np.ndarray           # (B,) broadside azimuth
    M: int
    K: int                      # users per BS
    h: np.ndarray               # (B_tx, B_usr, K, M) channels
    P: float = 1.0
    noise: float = 1.0
    alpha: np.ndarray = None    # (B, K) user weights
    snr_ref: float = 16.0       # echo SNR at R_REF for unit beam gain
    blocked: callable = None    # blocked(b, xy) -> bool (known map)

    def steer_to(self, b, xy):
        d = xy - self.bs_pos[b]
        phi = np.arctan2(d[1], d[0]) - self.bs_az[b]
        return steer(self.M, phi)

    def c_coef(self, b, xy):
        if self.blocked is not None and self.blocked(b, xy):
            return 0.0
        r = max(np.linalg.norm(xy - self.bs_pos[b]), 1.0)
        return self.snr_ref * (R_REF / r) ** 4


@dataclass
class Horizon:
    """Per-solve data: sensing coefficients along the predicted trajectory."""
    J0: list                    # per target: current posterior information (4x4)
    a: np.ndarray               # (H, B, Q, M) steering vectors
    c: np.ndarray               # (H, B, Q) sensing coefficients
    E: np.ndarray               # (H, B, Q, 4, 4) unit-SNR Fisher matrices
    F: np.ndarray
    Qn: list                    # per-target process-noise covariance
    prior_free: bool = False    # snapshot baseline: J[tau] = eps I + J_obs
    crit: str = "D"             # "D": log det J ; "A": -log tr(S J^-1 S^T) (position)


def build_horizon(scene, mus, Ps, dt, sigma_a, H, prior_free=False, crit="D"):
    sig = np.broadcast_to(np.asarray(sigma_a, float), (len(mus),))
    F, Qn = cv_F(dt), [cv_Q(dt, s) for s in sig]
    B, Qt, M = len(scene.bs_pos), len(mus), scene.M
    a = np.zeros((H, B, Qt, M), complex)
    c = np.zeros((H, B, Qt))
    E = np.zeros((H, B, Qt, 4, 4))
    for q in range(Qt):
        x = mus[q].copy()
        for tau in range(H):
            x = F @ x
            for b in range(B):
                a[tau, b, q] = scene.steer_to(b, x[:2])
                c[tau, b, q] = scene.c_coef(b, x[:2])
                Hm = H_mono(x, scene.bs_pos[b])
                E[tau, b, q] = Hm.T @ SINV @ Hm
    J0 = [np.linalg.inv(P) for P in Ps]
    return Horizon(J0, a, c, E, F, Qn, prior_free, crit)


# ------------------------------------------------------------ utilities
def gains(scene, hz, W):
    """g[tau,b,q] = (M/P) sum_k |a^H w|^2."""
    proj = np.einsum('tbqm,tbkm->tbqk', hz.a.conj(), W)
    return (scene.M / scene.P) * np.sum(np.abs(proj) ** 2, axis=-1)


def phi_pred(J, F, Qn):
    return np.linalg.inv(F @ np.linalg.inv(J) @ F.T + Qn)


def sensing(scene, hz, W, grad=False):
    """U_sen (sum over tau, q of logdet J) and dU/dW* if requested."""
    H, B, Qt = hz.c.shape
    g = gains(scene, hz, W)
    U = 0.0
    Js = np.zeros((Qt, H, 4, 4))
    for q in range(Qt):
        J = hz.J0[q]
        for tau in range(H):
            Jobs = np.einsum('b,bij->ij', hz.c[tau, :, q] * g[tau, :, q], hz.E[tau, :, q])
            J = (1e-6 * np.eye(4) if hz.prior_free else phi_pred(J, hz.F, hz.Qn[q])) + Jobs
            Js[q, tau] = J
            if hz.crit == "A":
                U -= np.log(np.trace(np.linalg.inv(J)[:2, :2]))
            else:
                U += np.linalg.slogdet(J)[1]
    if not grad:
        return U, None
    G = np.zeros((H, B, Qt))                 # dU/d g[tau,b,q]
    for q in range(Qt):
        Gam = np.zeros((4, 4))
        for tau in range(H - 1, -1, -1):
            Ji = np.linalg.inv(Js[q, tau])
            if hz.crit == "A":
                SJ = Ji[:, :2]
                Gam = SJ @ SJ.T / np.trace(Ji[:2, :2]) + Gam
            else:
                Gam = Ji + Gam
            G[tau, :, q] = hz.c[tau, :, q] * np.einsum('ij,bji->b', Gam, hz.E[tau, :, q])
            if tau > 0 and not hz.prior_free:
                Jp = np.linalg.inv(Js[q, tau - 1])
                S = hz.F @ Jp @ hz.F.T + hz.Qn[q]
                A = np.linalg.solve(S, hz.F @ Jp)
                Gam = A.T @ Gam @ A
            else:
                Gam = np.zeros((4, 4))
    # dg/dW* = (M/P) a a^H w
    proj = np.einsum('tbqm,tbkm->tbqk', hz.a.conj(), W)
    dW = (scene.M / scene.P) * np.einsum('tbq,tbqm,tbqk->tbkm', G, hz.a, proj)
    return U, dW


def comm(scene, W, wmmse=False):
    """Sum of alpha*log2(1+SINR) over slots and users; WMMSE auxiliaries."""
    h = scene.h                                           # (bt, bu, k, m)
    # y[tau, bt, j, bu, k] = h[bt,bu,k]^H W[tau,bt,j]
    y = np.einsum('xukm,txjm->txjuk', h.conj(), W)
    pw = np.abs(y) ** 2
    S = pw.sum(axis=(1, 2)) + scene.noise                 # (tau, bu, k)
    B, K = h.shape[1], h.shape[2]
    des = np.stack([y[:, b, :, b, :].diagonal(axis1=1, axis2=2) for b in range(B)], 1)  # (tau,bu,k)
    sinr = np.abs(des) ** 2 / (S - np.abs(des) ** 2)
    U = np.sum(scene.alpha[None] * np.log2(1 + sinr))
    if not wmmse:
        return U, sinr
    u = des / S                                           # MMSE receivers
    v = 1 + sinr
    return U, sinr, u, v


def project_ball(W, P):
    n = np.sqrt(np.sum(np.abs(W) ** 2, axis=(2, 3), keepdims=True))
    return W * np.minimum(1.0, np.sqrt(P) / np.maximum(n, 1e-300))


@dataclass
class SolveLog:
    f: list = field(default_factory=list)
    rho: list = field(default_factory=list)
    resid: list = field(default_factory=list)
    trials: list = field(default_factory=list)


def objective(scene, hz, W, eta):
    H = W.shape[0]
    Uc, _ = comm(scene, W)
    Us, _ = sensing(scene, hz, W)
    return eta * Uc / H + (1 - eta) * Us / H, Uc / H, Us / H


def solve(scene, hz, W0, eta, n_iter=50, rho0=0.1, tol=1e-6, log=None, relax=True):
    """Proximal SCA-WMMSE with certified adaptive rho (Algorithm 2)."""
    H = W0.shape[0]
    B, K, M = W0.shape[1], W0.shape[2], W0.shape[3]
    h = scene.h
    alt = scene.alpha / np.log(2) / H                     # natural-log weights
    W = project_ball(W0.copy(), scene.P)
    rho = rho0
    f, _, _ = objective(scene, hz, W, eta)
    if log is not None:
        log.f.append(f); log.rho.append(rho)
    for it in range(n_iter):
        Uc, sinr, u, v = comm(scene, W, wmmse=True)
        Us, q = sensing(scene, hz, W, grad=True)
        q = q / H
        # comm gradient (exact at W): eta*[alt v u h - sum alt v |u|^2 h h^H w]
        coef = alt[None] * v * np.abs(u) ** 2              # (tau,bu,k)
        Phi0 = np.einsum('tuk,xukm,xukn->txmn', coef, h, h.conj())   # (tau,bt,M,M)
        lin = np.zeros_like(W)
        for b in range(B):
            lin[:, b] = (alt[b][None, :, None] * (v[:, b] * u[:, b])[:, :, None]
                         * h[b, b][None])
        gradc = lin - np.einsum('txmn,txjn->txjm', Phi0, W)
        grad = eta * gradc + (1 - eta) * q                 # df/dW*
        # stationarity residual (step 1 / unit scaling)
        Wp = project_ball(W + 2 * grad, scene.P)
        resid = float(np.linalg.norm(Wp - W))
        if relax:                     # allow rho to relax; the test re-certifies it
            rho = max(rho0, rho / 2)
        trials = 0
        while True:
            trials += 1
            Wn = np.zeros_like(W)
            for tau in range(H):
                for b in range(B):
                    Phi = eta * Phi0[tau, b] + rho * np.eye(M)
                    C = (eta * lin[tau, b] + (1 - eta) * q[tau, b] + rho * W[tau, b]).T  # (M,K)
                    lam, U_ = np.linalg.eigh(Phi)
                    D = U_.conj().T @ C
                    pw = lambda mu: np.sum(np.abs(D) ** 2 / (lam[:, None] + mu) ** 2)
                    if pw(0.0) <= scene.P:
                        mu = 0.0
                    else:
                        lo, hi = 0.0, 1.0
                        while pw(hi) > scene.P:
                            hi *= 2
                        for _ in range(60):
                            mid = 0.5 * (lo + hi)
                            lo, hi = (mid, hi) if pw(mid) > scene.P else (lo, mid)
                        mu = hi
                    Wn[tau, b] = (U_ @ (D / (lam[:, None] + mu))).T
            # minorant value g(Wn; W, rho)
            y = np.einsum('xukm,txjm->txjuk', h.conj(), Wn)
            Sn = (np.abs(y) ** 2).sum(axis=(1, 2)) + scene.noise
            desn = np.stack([y[:, b, :, b, :].diagonal(axis1=1, axis2=2) for b in range(B)], 1)
            e = np.abs(u) ** 2 * Sn - 2 * np.real(u.conj() * desn) + 1
            Uhat = np.sum(alt[None] * (np.log(v) - v * e + 1))
            Usen_lin = Us / H + 2 * np.real(np.sum(q.conj() * (Wn - W)))
            gval = eta * Uhat + (1 - eta) * Usen_lin - rho * np.sum(np.abs(Wn - W) ** 2)
            fn, _, _ = objective(scene, hz, Wn, eta)
            if fn >= gval - 1e-12 * max(1, abs(gval)):
                break
            if trials > 60:                 # never reached in the experiments
                raise RuntimeError("minorization test not certified after 60 doublings")
            rho *= 2
        W = Wn
        if log is not None:
            log.f.append(fn); log.rho.append(rho); log.resid.append(resid)
            log.trials.append(trials)
        if abs(fn - f) <= tol * max(1.0, abs(f)):
            f = fn
            break
        f = fn
    return W, f


def mrt_init(scene, H):
    B, K, M = scene.h.shape[0], scene.K, scene.M
    W = np.zeros((H, B, K, M), complex)
    for b in range(B):
        for k in range(K):
            hb = scene.h[b, b, k]
            W[:, b, k] = np.sqrt(scene.P / K) * hb / np.linalg.norm(hb)
    return W
