"""
Experiment G: two-agent cooperative tracking on V2X-Radar-C (all 40 scenes).

Real ingredients: vehicle trajectories (annotated 3-D boxes, persistent IDs),
the stationary RSU and the moving ego vehicle poses, and per-agent visibility
(a vehicle is observable by an agent in a frame iff it appears in that agent's
annotation list). Measurements [AoA, range, range-rate] are synthesised from the
annotated positions with the CRB-SNR law; radar point clouds are not used.

Coordinates: in V2X-Radar-C both agents' labels are already expressed in the
common (RSU) frame -- the RSU lidar_pose is the identity and the two agents'
labels of the same vehicle agree to ~0.1-0.4 m -- so no pose transform is
applied to label locations. lidar_pose = [x, y, z, roll, yaw, pitch] gives the
ego position in the same frame.

The ego radar's range rate is taken as ego-motion compensated (its own
velocity is known from odometry), so h_mono with absolute velocity applies.

Per target we run: RSU-only EKF, ego-only EKF, centralised two-agent EKF,
distributed fusion with 8-bit anchored messages, and compute the snapshot
(K=1) and short-window (K=5) observability Gramians to test Theorem 1/Lemma 1.
"""
import json
import yaml
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import isac_core as ic
from isac_core import BS, FusionCfg, run_trackers, rmse, full_adj, window_gramian, coop_bound

ROOT = Path(__file__).resolve().parent.parent / "data/v2x_radar/V2X-Radar-C"
OUT = Path(__file__).resolve().parent.parent / "results" / "exp_G.json"
DT = 0.1
MIN_LEN = 60          # frames (6 s) of continuous union visibility
SEEDS = range(5)
ic.SNR_MAX = 100.0    # automotive-radar resolution floor (sigma_r >= 0.1 m)
SNR_REF = 1.0


def load_scene(sdir):
    frames = sorted(f.stem for f in (sdir / "-1").glob("*.yaml"))
    rows = []
    for fr in frames:
        a = yaml.safe_load(open(sdir / "-1" / f"{fr}.yaml"))
        b = yaml.safe_load(open(sdir / "142" / f"{fr}.yaml"))
        rsu, ego = (a, b) if a.get("RSU", False) else (b, a)
        rows.append(dict(rsu_pose=np.array(rsu["lidar_pose"], float),
                         ego_pose=np.array(ego["lidar_pose"], float),
                         rsu=rsu.get("vehicles") or {}, ego=ego.get("vehicles") or {}))
    return rows


def extract_targets(rows):
    """Continuous segments where at least one agent annotates the vehicle."""
    ids = set()
    for r in rows:
        ids |= set(r["rsu"]) | set(r["ego"])
    out, consist = [], []
    for vid in ids:
        T = len(rows)
        pos = np.full((T, 2), np.nan)
        vis = np.zeros((T, 2), bool)
        for t, r in enumerate(rows):
            locs = []
            if vid in r["rsu"]:
                locs.append(r["rsu"][vid]["location"][:2]); vis[t, 0] = True
            if vid in r["ego"]:
                locs.append(r["ego"][vid]["location"][:2]); vis[t, 1] = True
            if len(locs) == 2:
                consist.append(np.linalg.norm(np.subtract(locs[0], locs[1])))
            if locs:
                pos[t] = np.mean(locs, 0)
        ok = np.isfinite(pos[:, 0])
        # longest run of union visibility
        best, cur, start, bs = 0, 0, 0, 0
        for t in range(T):
            cur = cur + 1 if ok[t] else 0
            if cur == 1:
                start = t
            if cur > best:
                best, bs = cur, start
        if best >= MIN_LEN:
            sl = slice(bs, bs + best)
            p = pos[sl]
            # light smoothing of annotation jitter, then finite-difference velocity
            k = np.ones(3) / 3
            ps = np.column_stack([np.convolve(np.pad(p[:, i], 1, mode="edge"), k, "valid")
                                  for i in range(2)])
            v = np.gradient(ps, DT, axis=0)
            out.append(dict(vid=vid, traj=np.column_stack([ps, v]), vis=vis[sl],
                            ego=np.array([r["ego_pose"][:2] for r in rows])[sl],
                            rsu=rows[0]["rsu_pose"][:2]))
    return out, consist


def id_jumps(rows):
    """Frame-to-frame displacement of each (agent, object id): checks ID persistence."""
    out = []
    for ag in ("rsu", "ego"):
        for t in range(1, len(rows)):
            for k, v in rows[t][ag].items():
                if k in rows[t - 1][ag]:
                    out.append(float(np.linalg.norm(np.subtract(v["location"][:2], rows[t - 1][ag][k]["location"][:2]))))
    return out


def scene_job(sdir):
    rows = load_scene(sdir)
    tgts, consist = extract_targets(rows)
    jumps = id_jumps(rows)
    res = []
    for tg in tgts:
        tr, vis = tg["traj"], tg["vis"]
        T = len(tr)
        # ego moves: use its time-varying position by re-anchoring per step.
        # run_trackers takes static BSs, so we pass the ego trajectory via a
        # moving-BS wrapper: positions are updated inside a custom loop below.
        rec = dict(scene=sdir.name, vid=int(tg["vid"]), T=T,
                   frac_both=float(np.mean(vis.all(1))),
                   frac_rsu_only=float(np.mean(vis[:, 0] & ~vis[:, 1])),
                   frac_ego_only=float(np.mean(~vis[:, 0] & vis[:, 1])),
                   mean_range=[float(np.mean(np.linalg.norm(tr[:, :2] - tg["rsu"], axis=1))),
                               float(np.mean(np.linalg.norm(tr[:, :2] - tg["ego"], axis=1)))])
        out = moving_run(tr, tg["rsu"], tg["ego"], vis)
        rec.update(out)
        # Theorem 1 / Lemma 1 on snapshot (K=1) and short (K=5) windows
        th = []
        for t in range(5, T, 5):
            if not vis[t].all():
                continue
            bsr = [BS(tg["rsu"], SNR_REF)]
            bse = lambda tau, e=tg["ego"]: [BS(e[tau], SNR_REF)]
            for K in (1, 5):
                Gr = window_gramian(tr, t, K, bsr, DT, vis[:, :1])
                Ge = window_gramian(tr, t, K, bse, DT, vis[:, 1:])
                lam = [np.linalg.eigvalsh(G)[0] for G in (Gr, Ge, Gr + Ge)]
                best = 0 if lam[0] >= lam[1] else 1
                A, Bm = (Gr, Ge) if best == 0 else (Ge, Gr)
                th.append(dict(K=K, lam_r=float(lam[0]), lam_e=float(lam[1]),
                               lam_c=float(lam[2]), bound=float(coop_bound(A, Bm)),
                               rank_r=int(np.linalg.matrix_rank(Gr, tol=1e-9 * np.abs(Gr).max())),
                               rank_e=int(np.linalg.matrix_rank(Ge, tol=1e-9 * np.abs(Ge).max()))))
        rec["thm1"] = th
        res.append(rec)
    return res, (consist, len(rows), jumps)


def moving_run(tr, rsu_xy, ego_xy, vis):
    """Same as run_trackers but with a moving ego agent (B = 2)."""
    out = {}
    T = len(tr)
    for seed in SEEDS:
        rng_z = np.random.default_rng(1000 + seed)
        Z = np.full((T, 2, 3), np.nan); Rs = np.zeros((T, 2, 3, 3))
        bs_t = [[BS(rsu_xy, SNR_REF), BS(ego_xy[t], SNR_REF)] for t in range(T)]
        for t in range(T):
            for b in range(2):
                R = ic.R_meas(bs_t[t][b], tr[t]); Rs[t, b] = R
                w = rng_z.normal(size=3)          # always drawn (common random numbers)
                if vis[t, b]:
                    z = ic.h_mono(tr[t], bs_t[t][b].pos) + w * np.sqrt(np.diag(R))
                    z[0] = ic.wrap(z[0]); Z[t, b] = z
        F, Q = ic.cv_F(DT), ic.cv_Q(DT, 3.0)
        rng0 = np.random.default_rng(2000 + seed)
        mu0 = tr[0] + rng0.normal(size=4) * np.array([1, 1, 1, 1])
        P0 = np.diag([4.0, 4.0, 4.0, 4.0])
        for name, use, quant in (("rsu", [0], None), ("ego", [1], None),
                                 ("cen", [0, 1], None), ("dist8", [0, 1], 8),
                                 ("dist6", [0, 1], 6)):
            rq = np.random.default_rng(3000 + seed)
            nB = 2 if quant else 1
            mus = [mu0.copy() for _ in range(nB)]; Ps = [P0.copy() for _ in range(nB)]
            e2 = np.zeros(T)
            for t in range(T):
                for k in range(nB):
                    mus[k] = F @ mus[k]; Ps[k] = F @ Ps[k] @ F.T + Q
                if quant is None:
                    L = np.linalg.inv(Ps[0]); n = np.zeros(4)
                    for b in use:
                        Lb, nb = ic.local_info(mus[0], Z[t, b], bs_t[t][b], Rs[t, b])
                        L += Lb; n += nb
                    Ps[0] = np.linalg.inv(L); mus[0] = mus[0] + Ps[0] @ n
                    e = mus[0][:2] - tr[t, :2]
                else:
                    c = mus[0].copy()
                    infos = [ic.local_info(mus[b], Z[t, b], bs_t[t][b]) for b in range(2)]
                    # B = 2 on one link: one exact averaging round of the quantized messages
                    mus, Ps, _ = ic.dist_fuse(mus, Ps, infos, c, ic.metropolis(full_adj(2)), 1, quant,
                                           0.0, rq, rule="plain", fusion="hcmci")
                    ee = [mus[b][:2] - tr[t, :2] for b in range(2)]
                    e = np.array(ee)
                e2[t] = float(np.mean(np.sum(np.atleast_2d(e) ** 2, axis=-1)))
            out.setdefault(name, []).append(rmse(e2, burn=10))
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    scenes = sorted(p for split in ("train", "validate", "test")
                    for p in (ROOT / split).iterdir() if p.is_dir() and (p / "-1").is_dir())
    print(len(scenes), "scenes", flush=True)
    with Pool(18) as pool:
        parts = pool.map(scene_job, scenes)
    recs = [r for p, _ in parts for r in p]
    nframes = [c[1] for _, c in parts]
    jumps = np.concatenate([np.array(c[2], float) for _, c in parts])
    jumps = jumps[np.isfinite(jumps)]
    consist = np.concatenate([np.array(c[0], float) for _, c in parts if len(c[0])])
    consist = consist[np.isfinite(consist)]   # a few annotations carry NaN boxes
    print(f"label consistency across agents: median {np.median(consist):.3f} m, "
          f"p90 {np.percentile(consist, 90):.3f} m (n={len(consist)})")
    print(f"{len(recs)} targets from {len(set(r['scene'] for r in recs))} scenes")
    for k in ("rsu", "ego", "cen", "dist8", "dist6"):
        v = np.array([r[k] for r in recs])
        print(f"  {k:6s} median RMSE {np.median(v):.3f}  mean {np.mean(v):.3f}")
    th = [d for r in recs for d in r["thm1"]]
    for K in (1, 5):
        d = [x for x in th if x["K"] == K]
        hold = np.mean([x["lam_c"] >= x["bound"] * (1 - 1e-9) for x in d])
        sing = np.mean([min(x["rank_r"], x["rank_e"]) < 4 for x in d])
        print(f"  K={K}: n={len(d)}, single-agent singular {sing:.2f}, bound holds {hold:.3f}")
    print(f"frames per scene {min(nframes)}-{max(nframes)} (total {sum(nframes)}); "
          f"ID jump median {np.median(jumps):.3f} m, >3 m {100 * np.mean(jumps > 3):.2f}%")
    OUT.write_text(json.dumps(dict(records=recs, consistency=dict(
        median=float(np.median(consist)), p90=float(np.percentile(consist, 90)), n=int(len(consist))),
        frames=dict(min=int(min(nframes)), max=int(max(nframes)), total=int(sum(nframes))),
        id_jump=dict(median=float(np.median(jumps)), frac_gt3=float(np.mean(jumps > 3)), n=int(len(jumps))))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
