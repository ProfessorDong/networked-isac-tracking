"""
Experiment I: distributed tracking on ten real Boreas drives.

Ground-truth vehicle trajectories (Applanix GNSS/INS, 10 Hz, first 60 s of
each drive) are tracked by B = 4 roadside BSs placed alternately 60 m left and
right of the route. Measurements are synthesised with the CRB-SNR law; the
target motion is real (stops, turns, accelerations), so the CV prior is
mismatched. Coordinates are the drive's local ENU frame (origin at the start),
so absolute-form messages see a range that grows with distance travelled
(Corollary 2). Line cooperation graph, I = 8 rounds.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from isac_core import BS, FusionCfg, run_trackers, rmse, line_adj, pcrb

ROOT = Path(__file__).resolve().parent.parent / "data/boreas"
OUT = Path(__file__).resolve().parent.parent / "results" / "exp_I.json"
DT, NSTEP, B = 0.1, 600, 4
SEQS = ["boreas-2020-11-26-13-58", "boreas-2021-01-26-11-22", "boreas-2021-11-14-09-47",
        "boreas-2020-12-18-13-44", "boreas-2021-01-19-15-08", "boreas-2021-02-02-14-07",
        "boreas-2021-04-08-12-44", "boreas-2021-04-29-15-55", "boreas-2021-08-05-13-34",
        "boreas-2021-09-02-11-42"]
SEEDS = range(8)


def load_traj(seq):
    p = pd.read_csv(ROOT / seq / "applanix" / "lidar_poses.csv")
    t = p["GPSTime"].to_numpy(float) * 1e-6        # microseconds -> s
    t -= t[0]
    g = np.arange(0, t[-1], DT)[:NSTEP]
    cols = ["easting", "northing", "vel_east", "vel_north"]
    return np.column_stack([np.interp(g, t, p[c].to_numpy(float)) for c in cols])


def place_bss(tr):
    picks = np.linspace(len(tr) * 0.1, len(tr) * 0.9, B).astype(int)
    out = []
    for i, k in enumerate(picks):
        d = tr[min(k + 5, len(tr) - 1), :2] - tr[max(k - 5, 0), :2]
        if np.linalg.norm(d) < 1e-3:
            d = np.array([1.0, 0.0])
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        out.append(BS(tr[k, :2] + (1 if i % 2 == 0 else -1) * 60.0 * n, 16.0))
    return out


CFGS = lambda: [FusionCfg("cen"),
                FusionCfg("I8", line_adj(B), 8),
                FusionCfg("I8_b8", line_adj(B), 8, 8),
                FusionCfg("I8_b10", line_adj(B), 8, 10),
                FusionCfg("I8_b12", line_adj(B), 8, 12),
                FusionCfg("I8_b16", line_adj(B), 8, 16),
                FusionCfg("I8_b8_abs", line_adj(B), 8, 8, 0.0, "absolute"),
                FusionCfg("I8_b12_abs", line_adj(B), 8, 12, 0.0, "absolute"),
                FusionCfg("I8_b16_abs", line_adj(B), 8, 16, 0.0, "absolute"),
                FusionCfg("I8_b8_p02", line_adj(B), 8, 8, 0.2),
                FusionCfg("I2", line_adj(B), 2)]


def job(args):
    seq, seed = args
    tr = load_traj(seq)
    bss = place_bss(tr)
    spd = {}
    out = run_trackers(tr, bss, DT, CFGS(), seed=seed, sigma_a=1.5, spd_counts=spd)
    res = {k: rmse(v) for k, v in out.items()}
    res["spd"] = spd                                   # SPD-safeguard rejections per config
    return seq, seed, res


def main():
    info = {}
    for s in SEQS:
        tr = load_traj(s)
        sp = np.linalg.norm(tr[:, 2:], axis=1)
        acc = np.linalg.norm(np.gradient(tr[:, 2:], DT, axis=0), axis=1)
        P = pcrb(tr, place_bss(tr), DT, sigma_a=1.5)
        info[s] = dict(mean_speed=float(sp.mean()), max_speed=float(sp.max()),
                       p95_acc=float(np.percentile(acc, 95)),
                       pcrb_rmse=float(np.sqrt(np.mean([np.trace(p[:2, :2]) for p in P[20:]]))),
                       frac_stopped=float(np.mean(sp < 0.5)))
    with Pool(18) as pool:
        runs = pool.map(job, [(s, k) for s in SEQS for k in SEEDS])
    agg = {}
    for seq, seed, r in runs:
        for k, v in r.items():
            agg.setdefault(seq, {}).setdefault(k, []).append(v)
    for s in SEQS:
        a = agg[s]
        print(s, f"v={info[s]['mean_speed']:.1f} pcrb={info[s]['pcrb_rmse']:.3f}",
              " ".join(f"{k}={np.median(v):.3f}" for k, v in a.items() if k != "spd"))
    import isac_core as ic
    OUT.write_text(json.dumps(dict(info=info, agg=agg, provenance=ic.provenance(__file__))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
