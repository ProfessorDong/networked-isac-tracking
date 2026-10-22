"""
Experiment S: network size (Theorem 2, Corollary 1 as B grows).

B in {4, 8, 16, 32, 64} BSs, placed uniformly at random in a disk of radius
400 m around a synthetic CV trajectory, cooperate over a random geometric graph
(connection radius chosen for an expected degree of 6; resampled until
connected). A new geometry and graph is drawn for every seed. Every transmission
is requantized to m in {8, 12} bits; links fail with p in {0, 0.2}. For a geometric grid
of round counts I we log the realised consensus error, the Theorem-2 bound
(realised quantizer steps), and the tracking RMSE against the centralised EKF on
the same measurements (common random numbers). Corollary 1 is evaluated with the
measured E0 and mean V of each run.
Theorem 2 is checked per time step (realised error / bound, averaged over steps):
the time average of the raw errors is dominated by the few steps in which a BS
close to the target saturates the SNR cap. The realised Theorem-3 margin omega_t
is logged to explain divergences.
"""
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from isac_core import (synth_trajectory, BS, metropolis, lambda_bar, FusionCfg,
                       run_trackers, rmse)

OUT = Path(__file__).resolve().parent.parent / "results" / "exp_S.json"
T, DT = 200, 0.1
TRAJ = synth_trajectory(T, DT)
CENTER = TRAJ[:, :2].mean(0)
B_GRID = [4, 8, 16, 32, 64]
P_GRID = [0.0, 0.2]
BITS_GRID = [8, 12]
I_GRID = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135, 200, 300, 450, 700, 1000, 1500, 2200]
SEEDS = range(5)
RADIUS, DEGREE, SNR_REF = 400.0, 6.0, 256.0


def geometry(B, seed):
    rng = np.random.default_rng(50_000 + 97 * B + seed)
    rc = RADIUS * np.sqrt(DEGREE / B)
    for attempt in range(1000):
        r = RADIUS * np.sqrt(rng.uniform(size=B)); th = rng.uniform(0, 2 * np.pi, B)
        pos = CENTER + np.column_stack([r * np.cos(th), r * np.sin(th)])
        D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        A = ((D < rc) & (D > 0)).astype(float)
        # connectivity: Laplacian second-smallest eigenvalue
        L = np.diag(A.sum(1)) - A
        if np.sort(np.linalg.eigvalsh(L))[1] > 1e-9:
            return [BS(p, SNR_REF, f"bs{i}") for i, p in enumerate(pos)], A, attempt
        if attempt % 100 == 99:
            rc *= 1.05
    raise RuntimeError("no connected graph")


def job(args):
    B, p, seed = args
    bss, A, _ = geometry(B, seed)
    W = metropolis(A)
    ev = np.sort(np.abs(np.linalg.eigvalsh(W)))[::-1]
    cfgs = [FusionCfg("cen")] + [FusionCfg(f"m{m}_I{I}", A, I, m, p) for m in BITS_GRID for I in I_GRID]
    out, dg = run_trackers(TRAJ, bss, DT, cfgs, seed=seed, diag=True)
    lam = lambda_bar(W, p)
    res = dict(B=B, p=p, seed=seed, sigma2=float(ev[1]), lam=float(lam),
               deg=float(A.sum(1).mean()), cen=rmse(out["cen"]))
    for m in BITS_GRID:
        for I in I_GRID:
            d = dg[f"m{m}_I{I}"]
            a, b = d["act"][20:], d["bnd"][20:]
            ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
            e = out[f"m{m}_I{I}"]
            res[f"m{m}_I{I}"] = dict(rmse=rmse(e) if np.all(np.isfinite(e)) else float("inf"),
                                     ratio=float(np.mean(a[ok] / b[ok])), act=float(np.mean(a)),
                                     bnd=float(np.mean(b)), dev0=float(np.mean(d["dev0"][20:])),
                                     V=float(np.mean(d["V"][20:])),
                                     omega_max=float(np.max(d["omega"][20:])),
                                     omega_med=float(np.median(d["omega"][20:])))
        E0 = res[f"m{m}_I1"]["dev0"]
        V = np.mean([res[f"m{m}_I{I}"]["V"] for I in I_GRID])
        res[f"m{m}_Istar"] = float(np.log(B * E0 * np.log(1 / lam) / V) / np.log(1 / lam))
    return res


def main():
    jobs = [(B, p, s) for B in B_GRID for p in P_GRID for s in SEEDS]
    jobs.sort(key=lambda j: -j[0])                  # big networks first (load balance)
    with Pool(18) as pool:
        runs = pool.map(job, jobs)
    for m in BITS_GRID:
        for B in B_GRID:
            for p in P_GRID:
                rs = [r for r in runs if r["B"] == B and r["p"] == p]
                ratio = max(r[f"m{m}_I{I}"]["ratio"] for r in rs for I in I_GRID)
                imin = [I_GRID[int(np.argmin([r[f"m{m}_I{I}"]["act"] for I in I_GRID]))] for r in rs]
                div = [(r["seed"], I) for r in rs for I in I_GRID if r[f"m{m}_I{I}"]["rmse"] > 3 * r["cen"]]
                om = [r[f"m{m}_I{I}"]["omega_max"] for r in rs for I in I_GRID]
                print(f"m={m} B={B:2d} p={p}: lam {np.mean([r['lam'] for r in rs]):.3f} "
                      f"I* {np.mean([r[f'm{m}_Istar'] for r in rs]):6.1f} argmin act {imin} "
                      f"max mean-step ratio {ratio:.3f} | omega_max median {np.median(om):.2e} "
                      f"| runs >3x cen: {len(div)}/{len(rs) * len(I_GRID)}", flush=True)
    OUT.write_text(json.dumps(dict(I_grid=I_GRID, B_grid=B_GRID, p_grid=P_GRID, bits=BITS_GRID, runs=runs)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
