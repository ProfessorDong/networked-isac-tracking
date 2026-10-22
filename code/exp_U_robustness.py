"""
Experiment U: robustness outside Assumption 1 / Theorem 3's common-prior setting.

B = 6 BSs on a 300-m ring (as Exp B), ring graph, m = 8 bits, marginal link-failure
probability 0.2 in every failure model:
  iid      symmetric i.i.d. failures (Assumption 1 holds; reference)
  burst    symmetric Gilbert-Elliott failures, mean burst 5 rounds; the link state
           persists across rounds AND across tracking updates (violates i.i.d.)
  oneway   each DIRECTED link fails i.i.d.; only the receiver notices and returns the
           weight to itself, so C(i) is row- but not column-stochastic (violates
           double stochasticity: the network average drifts)
  priors   no failures, but every BS starts from its own prior mean (non-common
           priors; Theorem 3 assumes a common prior)
We report the tracking RMSE relative to the centralised EKF (common random
numbers) and the consensus error against the nominal Theorem-2 bound.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import isac_core as ic
from isac_core import synth_trajectory, ring_bss, ring_adj, FusionCfg, run_trackers, rmse

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_U.json"
T, DT, B = 200, 0.1, 6
TRAJ = synth_trajectory(T, DT)
BSS = ring_bss(TRAJ[:, :2].mean(0), 300.0, B)
ADJ = ring_adj(B)
P, BURST, BITS = 0.2, 5.0, 8
I_GRID = [1, 2, 3, 5, 8, 13, 21, 34]
SEEDS = range(10)
MODELS = ["iid", "burst", "oneway", "priors"]
_SAMPLE_W = ic.sample_W                               # original symmetric i.i.d. sampler


def make_sampler(model, seed):
    rng = np.random.default_rng(7000 + seed)
    state = {}
    r_ = 1.0 / BURST
    q_ = P * r_ / (1 - P)

    def burst(W, p_fail, _rng):
        Wi = W.copy()
        for i in range(B):
            for j in range(i + 1, B):
                if W[i, j] <= 0:
                    continue
                bad = state.get((i, j))
                if bad is None:
                    bad = rng.random() < P                 # stationary start
                else:
                    bad = (rng.random() >= r_) if bad else (rng.random() < q_)
                state[(i, j)] = bad
                if bad:
                    w = W[i, j]; Wi[i, j] = Wi[j, i] = 0.0; Wi[i, i] += w; Wi[j, j] += w
        return Wi

    def oneway(W, p_fail, _rng):
        Wi = W.copy()
        for i in range(B):                                 # receiver i, sender j
            for j in range(B):
                if i != j and W[i, j] > 0 and rng.random() < P:
                    Wi[i, i] += Wi[i, j]; Wi[i, j] = 0.0
        return Wi
    return {"burst": burst, "oneway": oneway}.get(model)


def job(args):
    model, seed = args
    p = 0.0 if model == "priors" else P
    spread = (5.0, 5.0, 2.0, 2.0) if model == "priors" else None
    res = {}
    for I in I_GRID:
        # every configuration gets its own link process (same seed), so the Markov burst
        # state is not shared with, or interleaved by, the other round counts
        smp = make_sampler(model, seed)
        ic.sample_W = smp if smp is not None else _SAMPLE_W    # worker-local
        cfgs = [FusionCfg("cen"), FusionCfg(f"I{I}", ADJ, I, BITS, p)]
        out, dg = run_trackers(TRAJ, BSS, DT, cfgs, seed=seed, diag=True, init_spread=spread)
        res["cen"] = rmse(out["cen"])                       # identical in every call (same seed)
        d = dg[f"I{I}"]
        e = out[f"I{I}"]
        res[f"I{I}"] = dict(rmse=rmse(e) if np.all(np.isfinite(e)) else float("inf"),
                            act=float(np.mean(d["act"][20:])), bnd=float(np.mean(d["bnd"][20:])))
    return model, seed, res


def main():
    with Pool(18) as pool:
        runs = pool.map(job, [(m, s) for m in MODELS for s in SEEDS])
    agg = {}
    for model, seed, res in runs:
        a = agg.setdefault(model, {"cen": [], **{f"I{I}": dict(rmse=[], act=[], bnd=[]) for I in I_GRID}})
        a["cen"].append(res["cen"])
        for I in I_GRID:
            for k in ("rmse", "act", "bnd"):
                a[f"I{I}"][k].append(res[f"I{I}"][k])
    for model, a in agg.items():
        cen = np.mean(a["cen"])
        print(model.ljust(7), " ".join(f"I{I}:{np.median(a[f'I{I}']['rmse']) / cen:.3f}"
                                       f"({np.mean(a[f'I{I}']['act']) / np.mean(a[f'I{I}']['bnd']):.2f})"
                                       for I in I_GRID), flush=True)
    OUT.write_text(json.dumps(dict(I_grid=I_GRID, p=P, burst=BURST, bits=BITS, agg=agg,
                                   provenance=ic.provenance(__file__))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
