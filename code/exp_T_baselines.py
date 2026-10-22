"""
Experiment T: accuracy versus backhaul bits per node per update, against flooding
of quantized innovations (same measurements, common random numbers).

Scenario: B BSs on a 300-m ring around the Exp-B trajectory; cooperation graph
either the ring or the circulant graph C_B(1,2) (each BS linked to its two nearest
neighbours on each side). Link failure probability p per transmission.

Proposed: anchored HCMCI messages, compensating consensus, m bits per scalar, I rounds,
pow2 block ranges (signed 8-bit exponents, realizable), pipelined anchor. Bits per node per
update: I (d m + 5 x 8) + 4 x 32 / (1-p). Statistic: paired per-seed RMSE ratio to the
centralized EKF (median; failure = ratio > 3 or non-finite).

Flooding of quantized innovations (batch quantized KF of Msechu et al. 2008; m = 1
is the SOI-KF of Ribeiro et al. 2006): every BS quantizes each component of its
measurement with m bits relative to its OWN prediction, using the Lloyd-Max
thresholds of the unit Gaussian applied to the normalised innovation, and floods the
3 indices. All BSs quantize in parallel, so one update takes one network diameter
of hops. A receiver reconstructs the interval with ITS OWN prediction and updates
sequentially with the exact truncated-Gaussian conditional mean and variance under
its current Gaussian belief. Without loss and with a common prior all nodes stay
identical (the setting of [Ribeiro2006, Msechu2008]); a lost message (a sender
not connected to the receiver in that update's surviving graph) is NOT
retransmitted, so nodes desynchronise. Bits per node: 3 m per message relayed.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import norm
import isac_core as ic
from isac_core import (synth_trajectory, ring_bss, ring_adj, FusionCfg, run_trackers, rmse,
                       gen_measurements, cv_F, cv_Q, h_mono, H_mono, R_meas, wrap)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_T.json"
T, DT = 200, 0.1
TRAJ = synth_trajectory(T, DT)
SEEDS = range(8)
CONS_BITS = [4, 6, 8, 10]
I_GRID = [1, 2, 3, 5, 8, 13, 21, 34]
FLOOD_BITS = [1, 2, 3, 4, 6]
P_GRID = [0.0, 0.05, 0.2]
SETUPS = [("ring", 6), ("circ", 6), ("ring", 16), ("circ", 16), ("rr", 64), ("rr", 128)]
BIG = 16                      # for B > BIG, flooding is simulated without loss only (ARQ costing)


def circ_adj(B, hops=2):
    A = np.zeros((B, B))
    for i in range(B):
        for h in range(1, hops + 1):
            A[i, (i + h) % B] = A[(i + h) % B, i] = 1
    return A


def rr_adj(B, deg=6, seed=0):
    """Connected random deg-regular graph (configuration model with retries)."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        stubs = rng.permutation(np.repeat(np.arange(B), deg))
        e = stubs.reshape(-1, 2)
        if np.any(e[:, 0] == e[:, 1]):
            continue
        A = np.zeros((B, B))
        if len({tuple(sorted(x)) for x in e}) < len(e):
            continue
        A[e[:, 0], e[:, 1]] = A[e[:, 1], e[:, 0]] = 1
        L = np.diag(A.sum(1)) - A
        if np.sort(np.linalg.eigvalsh(L))[1] > 1e-9:
            return A
    raise RuntimeError("no regular graph")


def graph(kind, B):
    return {"ring": ring_adj, "circ": circ_adj, "rr": rr_adj}[kind](B)


def diameter(A):
    B = len(A)
    D = np.full((B, B), np.inf); np.fill_diagonal(D, 0)
    D[A > 0] = 1
    for k in range(B):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return int(D.max())


def lloyd_max(levels, iters=500):
    """Thresholds (len levels+1, with +-inf) of the Lloyd-Max quantizer of N(0,1)."""
    if levels == 2:
        return np.array([-np.inf, 0.0, np.inf])
    t = np.r_[-np.inf, np.linspace(-2.5, 2.5, levels - 1), np.inf]
    for _ in range(iters):
        a, b = t[:-1], t[1:]
        c = (norm.pdf(a) - norm.pdf(b)) / np.maximum(norm.cdf(b) - norm.cdf(a), 1e-300)
        t = np.r_[-np.inf, (c[:-1] + c[1:]) / 2, np.inf]
    return t


TH = {m: lloyd_max(2 ** m) for m in FLOOD_BITS}


def reach(A, p, rng):
    """Boolean matrix R[s, j]: message of s reaches j over the surviving links (one draw per message)."""
    B = len(A)
    R = np.zeros((B, B), bool)
    for s in range(B):
        alive = np.triu(A * (rng.random(A.shape) >= p), 1)
        alive = alive + alive.T
        seen = {s}; front = [s]
        while front:
            n = front.pop()
            for j in np.flatnonzero(alive[n]):
                if j not in seen:
                    seen.add(j); front.append(j)
        R[s, list(seen)] = True
    return R


def flood(traj, bss, A, seed, bits, p, sigma_a=1.0):
    rng_z = np.random.default_rng(1000 + seed)
    Z, _ = gen_measurements(traj, bss, rng_z, None)              # same draws as run_trackers
    rng_l = np.random.default_rng(5000 + seed)
    F, Q = cv_F(DT), cv_Q(DT, sigma_a)
    B = len(bss)
    rng0 = np.random.default_rng(2000 + seed)
    mu0 = traj[0] + rng0.normal(size=4) * np.array((2, 2, 1, 1))
    nN = B if p > 0 else 1                     # without loss all nodes stay identical
    mus = [mu0.copy() for _ in range(nN)]
    Ps = [np.diag([25.0, 25.0, 4.0, 4.0]) for _ in range(nN)]
    th = TH[bits]
    e2 = np.zeros(len(traj))
    for t in range(len(traj)):
        for j in range(nN):
            mus[j] = F @ mus[j]; Ps[j] = F @ Ps[j] @ F.T + Q
        # each sender quantizes its normalised innovations w.r.t. its own prediction
        idx = np.zeros((B, 3), int)
        for b, bs in enumerate(bss):
            sb = b if nN > 1 else 0
            y0 = h_mono(mus[sb], bs.pos); H = H_mono(mus[sb], bs.pos); R = R_meas(bs, mus[sb])
            for k in range(3):
                s0 = np.sqrt(H[k] @ Ps[sb] @ H[k] + R[k, k])
                r = Z[t, b, k] - y0[k]
                if k == 0:
                    r = wrap(r)
                idx[b, k] = int(np.searchsorted(th, r / s0) - 1)
        Rch = reach(A, p, rng_l) if p > 0 else np.ones((B, B), bool)
        for j in range(nN):
            mu, P = mus[j], Ps[j]
            for b, bs in enumerate(bss):
                if not Rch[b, j]:
                    continue
                # receiver's reconstruction of the interval: its own prediction
                y0 = h_mono(mus[j], bs.pos); H0 = H_mono(mus[j], bs.pos); R0 = R_meas(bs, mus[j])
                for k in range(3):
                    s0 = np.sqrt(H0[k] @ Ps[j] @ H0[k] + R0[k, k])
                    h = H_mono(mu, bs.pos)[k]
                    yc = h_mono(mu, bs.pos)[k]
                    d = wrap(y0[k] - yc) if k == 0 else y0[k] - yc   # interval offset w.r.t. yc
                    s = np.sqrt(h @ P @ h + R_meas(bs, mu)[k, k])
                    a_ = (d + s0 * th[idx[b, k]]) / s
                    b_ = (d + s0 * th[idx[b, k] + 1]) / s
                    Zp = norm.cdf(b_) - norm.cdf(a_)
                    if Zp < 1e-12:                                 # inconsistent interval (desync)
                        continue
                    fa = 0.0 if np.isinf(a_) else norm.pdf(a_)
                    fb = 0.0 if np.isinf(b_) else norm.pdf(b_)
                    afa = 0.0 if np.isinf(a_) else a_ * fa
                    bfb = 0.0 if np.isinf(b_) else b_ * fb
                    al = (fa - fb) / Zp
                    var = 1 + (afa - bfb) / Zp - al ** 2
                    g = P @ h / s
                    mu = mu + g * al
                    P = P - (1 - var) * np.outer(g, g)
            mus[j], Ps[j] = mu, P
        e2[t] = np.mean([np.sum((m[:2] - traj[t, :2]) ** 2) for m in mus])
    return rmse(e2) if np.all(np.isfinite(e2)) else float("inf")


def job(args):
    kind, B, p, seed = args
    bss = ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
    A = graph(kind, B)
    # realizable consensus protocol: pow2 block ranges sent as 8-bit exponents, pipelined
    # anchor (leader's posterior from two updates earlier, predicted forward) so that anchor
    # dissemination overlaps the previous update
    cfgs = [FusionCfg("cen")] + [FusionCfg(f"cons_m{m}_I{I}", A, I, m, p, "anchored", "compensating",
                                           "hcmci", "pow2", 1)
                                 for m in CONS_BITS for I in I_GRID]
    out = run_trackers(TRAJ, bss, DT, cfgs, seed=seed)
    res = {k: (rmse(v) if np.all(np.isfinite(v)) else float("inf")) for k, v in out.items()}
    for m in FLOOD_BITS:
        if p == 0 or B <= BIG:
            res[f"flood_m{m}"] = flood(TRAJ, bss, A, seed, m, p)
        res[f"floodarq_m{m}"] = flood(TRAJ, bss, A, seed, m, 0.0)   # lossless delivery via ARQ
    return kind, B, p, seed, res


def cost(name, B, diam, p):
    """(payload bits, total bits with 32-bit ranges, total with 8-bit ranges, latency in hops/rounds).
    Logical payload/hop model (no airtime, contention or scheduling). ARQ: every hop is repeated
    until it succeeds, 1/(1-p) transmissions and slots on average. Consensus is simulated with
    pow2 ranges, so 'bits8' is its realizable cost; the anchor (4 x 32 bits) is relayed by every
    node with ARQ during the previous update (pipelined: no added latency)."""
    if name.startswith("flood"):
        m = int(name.split("_m")[1])
        k = 1.0 / (1 - p) if name.startswith("floodarq") else 1.0
        b = B * 3 * m * k
        return b, b, b, diam * k
    _, m, I = name.split("_")
    m, I = int(m[1:]), int(I[1:])
    pay = I * ic.N_MSG * m
    anc = 4 * 32 / (1 - p)
    return pay, pay + I * 5 * 32 + anc, pay + I * 5 * 8 + anc, I


def main():
    jobs = [(k, B, p, s) for (k, B) in SETUPS for p in P_GRID for s in SEEDS]
    jobs.sort(key=lambda j: -j[1])
    with Pool(18) as pool:
        runs = pool.map(job, jobs)
    summary = {}
    (OUT.parent / "exp_T_runs.json").write_text(json.dumps(dict(runs=runs)))   # raw backup
    for kind, B in SETUPS:
        A = graph(kind, B); diam = diameter(A)
        lam = {str(p): ic.lambda_bar(ic.metropolis(A), p) for p in P_GRID}
        for p in P_GRID:
            rs = [r for k_, B_, p_, s_, r in runs if (k_, B_, p_) == (kind, B, p)]
            cen = float(np.mean([r["cen"] for r in rs]))
            c_s = np.array([r["cen"] for r in rs])
            pts = []
            for name in rs[0]:
                if name == "cen":
                    continue
                v = np.array([r[name] for r in rs])
                ratio = v / c_s                          # paired per-seed RMSE ratio
                pay, b32, b8, lat = cost(name, B, diam, p)
                pts.append(dict(name=name, family=name.split("_")[0],
                                payload=pay, bits32=b32, bits8=b8, latency=lat,
                                rel=float(np.median(ratio)),             # median paired ratio
                                rel_max=float(np.max(ratio)),
                                rel_pooled=float(np.sqrt(np.mean(v ** 2) / np.mean(c_s ** 2))),
                                frac_div=float(np.mean(~np.isfinite(ratio) | (ratio > 3)))))
            key = f"{kind}{B}_p{p}"
            summary[key] = dict(kind=kind, B=B, p=p, cen=cen, diam=diam, lam=lam[str(p)], points=pts)
            print(f"{key}: cen {cen:.4f} diam {diam} lam {lam[str(p)]:.3f}", flush=True)
            for fam, bk in (("flood", "bits32"), ("floodarq", "bits32"), ("cons", "bits32"), ("cons", "bits8")):
                best = np.inf
                for q in sorted([q for q in pts if q["family"] == fam], key=lambda q: q[bk]):
                    if q["rel"] < best - 1e-4:
                        best = q["rel"]
                        print(f"   {fam:5s}[{bk}] {q[bk]:7.0f} bits lat {q['latency']:5.1f}  rel {q['rel']:.3f} "
                              f"div {q['frac_div']:.2f}  ({q['name']})")
    OUT.write_text(json.dumps(dict(setups=SETUPS, p_grid=P_GRID, summary=summary,
                                   provenance=ic.provenance(__file__))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
