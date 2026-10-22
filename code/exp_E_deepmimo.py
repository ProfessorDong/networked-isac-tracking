"""
Experiment E: tracking over ray-traced echo-SNR maps (DeepMIMO v4).

Four outdoor multi-BS scenarios with three BSs each (NYC 3.5 GHz, Los Angeles
3.5 GHz, Phoenix 3.5 GHz, NYC 28 GHz). For BS b the one-way ray-traced path
gain g_b(x) (sum of path powers at the nearest grid point) defines the
monostatic echo SNR  SNR_b(x) = snr_ref * (g_b(x) / g_b,100)^2, where g_b,100
is the median one-way gain at 90-110 m from BS b; the square is the two-way
budget. A target is detectable by BS b iff SNR_b(x) >= 1 (0 dB), so the maps
create shadowed coverage holes. (The single-BS ASU-campus scenario and the
indoor i1 scenario, which fails to load in deepmimo 4.0.1, are not used.)

Per scenario and trajectory: single-BS EKFs, centralised EKF, distributed
fusion on a line graph (I = 5) at full precision and 8-16 bits (anchored),
and 12/16-bit absolute-frame messages.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from scipy.spatial import cKDTree
import isac_core as ic
from isac_core import BS, FusionCfg, run_trackers, rmse, line_adj, cv_F

DATA = Path(__file__).resolve().parent.parent / "data/deepmimo_scenarios"
OUT = Path(__file__).resolve().parent.parent / "results" / "exp_E.json"
SCEN = [("city_0_newyork_3p5_s", "NYC 3.5 GHz"), ("city_1_losangeles_3p5", "LA 3.5 GHz"),
        ("city_4_phoenix_3p5", "Phoenix 3.5 GHz"), ("city_0_newyork_28", "NYC 28 GHz")]
DT, T, SNR_REF = 0.1, 300, 256.0
SEEDS = range(12)


def cached_maps(name):
    f = DATA / f"_cache_{name}.npz"
    if not f.exists():
        import deepmimo as dm
        dm.config.set("scenarios_folder", str(DATA))
        s = dm.load(name)
        ds = s.datasets if hasattr(s, "datasets") else [s]
        tx = np.array([np.asarray(d.tx_pos)[0][:2] for d in ds])
        rx = np.asarray(ds[0].rx_pos)[:, :2]
        g = np.stack([np.nansum(np.where(np.isfinite(np.asarray(d.power)),
                                         10 ** (np.asarray(d.power) / 10), 0.0), 1) for d in ds])
        np.savez(f, tx=tx, rx=rx, g=g)
    z = np.load(f)
    return z["tx"], z["rx"], z["g"]


class MapBS(BS):
    pass


def make_bss(tx, rx, g):
    kd = cKDTree(rx)
    out = []
    for b in range(len(tx)):
        r = np.linalg.norm(rx - tx[b], axis=1)
        ring = (r > 90) & (r < 110) & (g[b] > 0)
        gref = np.median(g[b][ring]) if ring.sum() > 20 else np.median(g[b][g[b] > 0])
        bs = MapBS(tx[b].astype(float), SNR_REF, f"BS{b}")
        gb = g[b] / gref
        bs.snr_fn = (lambda x, gb=gb: float(SNR_REF * gb[kd.query(x[:2])[1]] ** 2))
        out.append(bs)
    return out


def trajectory(rx, g, seed):
    """Smooth 8 m/s path that stays on ray-reachable ('street') grid points:
    a step is accepted only if some BS has a propagation path to the nearest
    grid point; otherwise the heading is rotated (as a vehicle turns at a
    building edge)."""
    rng = np.random.default_rng(seed)
    kd = cKDTree(rx)
    street = (g > 0).any(0)
    lo, hi = rx.min(0) + 5, rx.max(0) - 5
    ok = lambda p: np.all(p >= lo) and np.all(p <= hi) and street[kd.query(p)[1]]
    cand = rx[street & np.all((rx > lo + 20) & (rx < hi - 20), 1)]
    x = np.r_[cand[rng.integers(len(cand))], 0, 0]
    th = rng.uniform(0, 2 * np.pi)
    out = [x.copy()]
    for t in range(1, T):
        th += 0.02 * rng.normal()
        for k in range(60):
            v = 8.0 * np.array([np.cos(th), np.sin(th)])
            if ok(x[:2] + DT * v):
                break
            th += (np.pi / 12) * (1 if (seed + t) % 2 else -1)
        x = np.r_[x[:2] + DT * v, v]
        out.append(x.copy())
    tr = np.array(out)
    k = np.ones(9) / 9                            # smooth the heading changes
    for i in range(2):
        tr[:, i] = np.convolve(np.pad(tr[:, i], 4, mode="edge"), k, "valid")
    tr[:, 2:] = np.gradient(tr[:, :2], DT, axis=0)
    return tr


def job(args):
    name, seed = args
    tx, rx, g = cached_maps(name)
    bss = make_bss(tx, rx, g)
    tr = trajectory(rx, g, seed)
    snr = np.array([[bs.snr_fn(x) for bs in bss] for x in tr])
    vis = snr >= 1.0
    B = len(bss)
    cfgs = [FusionCfg("cen"), FusionCfg("I5", line_adj(B), 5),
            FusionCfg("I5_b8", line_adj(B), 5, 8), FusionCfg("I5_b10", line_adj(B), 5, 10),
            FusionCfg("I5_b12", line_adj(B), 5, 12), FusionCfg("I5_b14", line_adj(B), 5, 14),
            FusionCfg("I5_b16", line_adj(B), 5, 16),
            FusionCfg("I5_b12_abs", line_adj(B), 5, 12, 0.0, "absolute"),
            FusionCfg("I5_b16_abs", line_adj(B), 5, 16, 0.0, "absolute")]
    spd, ov0 = {}, ic.CODEC_OVERFLOW[0]
    out = run_trackers(tr, bss, DT, cfgs, seed=seed, sigma_a=1.0, vis=vis, spd_counts=spd)
    res = {k: rmse(v) for k, v in out.items()}
    res["spd"] = spd                                   # SPD-safeguard rejections per config
    res["codec_overflow"] = ic.CODEC_OVERFLOW[0] - ov0
    single = []
    for b in range(B):
        vb = np.zeros_like(vis); vb[:, b] = vis[:, b]
        single.append(rmse(run_trackers(tr, bss, DT, [FusionCfg("s")], seed=seed,
                                        sigma_a=1.0, vis=vb)["s"]))
    res["best_single"] = float(np.min(single))
    res["cov_any"] = float(np.mean(vis.any(1)))
    res["cov_multi"] = float(np.mean(vis.sum(1) >= 2))
    sdb = 10 * np.log10(np.minimum(snr[vis], ic.SNR_MAX))      # as seen by the receivers
    res["snr_db_med"] = float(np.median(sdb))
    S = np.where(vis, 10 * np.log10(np.minimum(np.maximum(snr, 1e-12), ic.SNR_MAX)), np.nan)
    multi = vis.sum(1) >= 2                                      # SNR spread across BSs
    res["snr_db_spread"] = float(np.median(np.nanmax(S[multi], 1) - np.nanmin(S[multi], 1)))
    return name, seed, res


def main():
    for n, _ in SCEN:
        cached_maps(n)
    with Pool(18) as pool:
        runs = pool.map(job, [(n, s) for n, _ in SCEN for s in SEEDS])
    agg = {}
    for n, s, r in runs:
        for k, v in r.items():
            agg.setdefault(n, {}).setdefault(k, []).append(v)
    for n, lab in SCEN:
        a = agg[n]
        print(f"{lab:16s}", " ".join(f"{k}={np.median(v):.3g}" for k, v in a.items() if k != "spd"))
    OUT.write_text(json.dumps(dict(scenarios=SCEN, agg=agg, provenance=ic.provenance(__file__))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
