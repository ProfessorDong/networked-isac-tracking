"""Plot data for the TikZ/pgfplots figures, regenerated from results/exp_*.json.

The figures themselves are LaTeX files figs/fig_*.tex, drawn by pgfplots inside
the manuscript (vector, same fonts as the text). This script only writes the
whitespace-separated tables they read, figs/data/*.dat. Never edit those by hand.
Missing values are written as "nan" (pgfplots: unbounded coords=jump).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "figs" / "data"
DATA.mkdir(parents=True, exist_ok=True)


def load(n):
    return json.load(open(RES / f"exp_{n}.json"))


def write(name, cols: dict):
    """Write equal-length columns to figs/data/<name>.dat."""
    keys = list(cols)
    arr = np.column_stack([np.asarray(cols[k], float) for k in keys])
    with open(DATA / f"{name}.dat", "w") as f:
        f.write(" ".join(keys) + "\n")
        for row in arr:
            f.write(" ".join("nan" if not np.isfinite(v) else f"{v:.6g}" for v in row) + "\n")
    print("wrote", name, arr.shape)


def pos(x):
    """Log axes: non-positive values become gaps."""
    x = np.asarray(x, float).copy()
    x[~(x > 0)] = np.nan
    return x


# --------------------------------------------------------------- Exp B
def data_B():
    d = load("B")
    I = np.array(d["I_grid"])
    ag, meta = d["agg"], d["meta"]
    lam = meta["ring"]["lam"]["0.0"]; kap = meta["ring"]["rho"]["0.0"]
    key = lambda topo, m, p=0.0, rule="compensating": f"{topo}_b{m}_p{p}_{rule}_hcmci"

    def curve(name, k):
        a = ag[k]
        act = np.array([np.mean(a[f"I{i}"]["act"]) for i in I])
        bnd = np.array([np.mean(a[f"I{i}"]["bnd"]) for i in I])
        ok = act > 1e-18 * act[0]                 # drop the double-precision floor
        write(name, dict(I=I, act=np.where(ok, act, np.nan), bnd=np.where(ok, bnd, np.nan)))
        return act, bnd

    for t in ("ring", "line"):
        curve(f"B_a_{t}", key(t, None))
    star = dict(m=[], I=[], act=[])
    eps = dict(m=[], I=[], act=[])
    for b in (4, 6, 8):
        actc, _ = curve(f"B_b_m{b}", key("ring", b))
        actp, _ = curve(f"B_b_plain_m{b}", key("ring", b, rule="plain"))
        ap, ac = ag[key("ring", b, rule="plain")], ag[key("ring", b)]
        # Corollary 1 with the measured E0 and mean V of each run family
        E0 = np.mean(ap["I1"]["dev0"]); V = np.mean([np.mean(ap[f"I{i}"]["V"]) for i in I])
        istar = np.log(d["B"] * E0 * np.log(1 / lam) / V) / np.log(1 / lam)
        if istar >= 1:
            k = int(np.argmin(np.abs(I - istar)))
            star["m"].append(b); star["I"].append(I[k]); star["act"].append(actp[k])
        E0c = np.mean(ac["I1"]["dev0"]); Vc = np.mean([np.mean(ac[f"I{i}"]["V"]) for i in I])
        ie = np.log((1 - lam) * E0c / (kap * Vc)) / np.log(1 / lam)
        k = int(np.argmin(np.abs(I - ie)))
        eps["m"].append(b); eps["I"].append(I[k]); eps["act"].append(actc[k])
    write("B_b_istar", star)
    write("B_b_ieps", eps)
    for p in ("0.0", "0.2", "0.4"):
        curve(f"B_c_p{p}", key("ring", 8, p))


# --------------------------------------------------------------- Exp D
def data_D():
    d = load("D")
    bits = np.array(d["bits"])
    for key in ("anchored_0", "absolute_0", "absolute_1000", "absolute_10000"):
        a = d["agg"][key]
        emp, pred = [], []
        for b in bits:
            m = np.array(a[f"b{b}"]["dev"]); mse = np.array(a[f"b{b}"]["mse"])
            # unconditional mean over ALL seeds, shown only if no seed fails (tracking MSE
            # > 1e2 m^2, the declared criterion) and the realised margin is < 1 in every seed
            ok = (np.all(np.isfinite(m)) and np.all(mse <= 1e2)
                  and np.max(a[f"b{b}"]["omega_max"]) < 1)
            emp.append(np.mean(m) if ok else np.nan)
            pred.append(np.mean(a[f"b{b}"]["pred_exc"]))
        emp, pred = np.array(emp), np.array(pred)
        pred[bits < 3] = np.nan
        write(f"D_a_{key}", dict(m=bits, pred=pos(pred), emp=pos(emp)))

    def need(key, which):
        a = d["agg"][key]; cen = np.array(a["cen_mse"]); thr = 0.01 * np.mean(cen)
        for b in bits:
            m = np.array(a[f"b{b}"]["dev"])
            v = (np.mean(m) if np.all(np.isfinite(m)) else np.inf) if which == "emp" \
                else np.mean(a[f"b{b}"]["pred_exc"])
            if v < thr:
                return b
        return np.nan
    for form in ("absolute", "anchored"):
        keys = [f"{form}_{o}" for o in (0, 1000, 10000)]          # every cell simulated
        write(f"D_b_{form}", dict(x=[0, 1, 2], pred=[need(k, "pred") for k in keys],
                                  emp=[need(k, "emp") for k in keys]))


# --------------------------------------------------------------- Exp J
def data_J():
    d = load("J")
    rho = np.array(d["rhos"]); r = d["rows"]; m0 = d["mse0"]
    x = np.linspace(0, rho.max(), 100)
    write("J_theory", dict(kappa=x, dyn=m0 + x ** 2 * d["lam_dyn"], static=m0 + x ** 2 * d["lam_static"]))
    write("J_sim", dict(kappa=rho, naive_wc=[r[f"naive_wc_{p}"][0] for p in rho],
                        naive_rand=[r[f"naive_rand_{p}"][0] for p in rho],
                        joint_wc=[r[f"joint_wc_{p}"][0] for p in rho]))


# --------------------------------------------------------------- Exp A
def data_A():
    d = load("A")
    for key in ("D_H1_eta0.5", "D_H4_eta0.5", "A_H4_eta0.5"):
        r = np.array(d["convergence"][key]["resid"])
        write(f"A_a_{key.replace('_eta0.5', '')}", dict(t=np.arange(1, len(r) + 1), resid=pos(r)))
    runs = [r for r in d["runs"] if "fus" not in r]       # integrated-loop runs are reported in the text
    for m in ("mpcA", "myopicA", "mpcD", "myopicD", "snapshot", "snapshotA"):
        pts = {}
        for r in runs:
            if r["method"] != m:
                continue
            e = np.array(r["err2"])
            pts.setdefault(r["eta"], []).append((np.mean(r["rate"]), np.sqrt(e[10:].mean())))
        etas = sorted(pts)
        R = [np.mean([p[0] for p in pts[e]]) for e in etas]
        E = [np.mean([p[1] for p in pts[e]]) for e in etas]
        S = [1.96 * np.std([p[1] for p in pts[e]], ddof=1) / np.sqrt(len(pts[e])) for e in etas]
        write(f"A_b_{m}", dict(eta=etas, rate=R, rmse=E, ci=S))
    cm = [r for r in runs if r["method"] == "comm"]
    rc = np.mean([np.mean(r["rate"]) for r in cm])           # communication-only sum rate
    ec = np.mean([np.sqrt(np.array(r["err2"])[10:].mean()) for r in cm])
    write("A_b_commline", dict(gap=[1.5e-3, 40.0], rmse=[ec, ec]))   # horizontal reference
    for m in ("mpcA", "myopicA", "mpcD", "myopicD", "snapshot", "snapshotA"):
        t = np.genfromtxt(DATA / f"A_b_{m}.dat", names=True)
        write(f"A_b_{m}", dict(eta=t["eta"], rate=t["rate"], gap=rc - t["rate"], rmse=t["rmse"], ci=t["ci"]))
    write("A_b_comm", dict(rate=[np.mean([np.mean(r["rate"]) for r in cm])],
                           rmse=[np.mean([np.sqrt(np.array(r["err2"])[10:].mean()) for r in cm])]))


# --------------------------------------------------------------- Exp G
def data_G():
    d = load("G")
    recs = d["records"]
    th = [x for r in recs for x in r["thm1"]]
    k1 = [x for x in th if x["K"] == 1]
    write("G_a_snap", dict(lam=pos([x["lam_c"] for x in k1]), y=pos([x["bound"] for x in k1])))
    k5 = [x for x in th if x["K"] == 5]
    write("G_a_win", dict(lam=pos([x["lam_c"] for x in k5]),
                          y=pos([max(x["lam_r"], x["lam_e"]) for x in k5])))
    for k in ("rsu", "ego", "cen", "dist8"):
        v = np.sort(np.array([r[k] for r in recs]))
        v = v[np.isfinite(v)]
        write(f"G_b_{k}", dict(rmse=v, cdf=np.arange(1, len(v) + 1) / len(recs)))


# --------------------------------------------------------------- Exp T
def data_T():
    d = load("T")["summary"]

    def frontier(key, fam, bk):
        pts = sorted([q for q in d[key]["points"] if q["family"] == fam and q["frac_div"] == 0],
                     key=lambda q: q[bk])
        xs, ys, best = [], [], np.inf
        for q in pts:
            if q["rel"] < best - 1e-6:
                best = q["rel"]; xs.append(q[bk]); ys.append(q["rel"])
        return xs, ys
    for key, tag in (("ring16_p0.0", "ring16"), ("rr128_p0.0", "rr128"), ("ring16_p0.2", "ring16loss"),
                     ("circ16_p0.2", "circ16loss")):
        for fam, bk in (("flood", "bits32"), ("floodarq", "bits32"), ("cons", "bits8")):
            if not any(q["family"] == fam for q in d[key]["points"]):
                continue
            xs, ys = frontier(key, fam, bk)
            if xs:
                write(f"T_{tag}_{fam}", dict(bits=xs, rel=ys))
        if "loss" in tag:                         # flooding without ARQ: every bit width
            fl = sorted([q for q in d[key]["points"] if q["family"] == "flood"], key=lambda q: q["bits32"])
            write(f"T_{tag}_floodall", dict(bits=[q["bits32"] for q in fl], rel=[q["rel"] for q in fl]))


# --------------------------------------------------------------- Exp S, U, R (supplement)
def data_S():
    import exp_S_network_scale as S
    from isac_core import metropolis, rho_bar
    d = load("S"); G = np.array(d["I_grid"])
    ie = dict(B=[], I=[])
    for B in d["B_grid"]:
        rs = [r for r in d["runs"] if r["B"] == B and r["p"] == 0.0]
        rel = np.array([[r[f"m12_I{i}"]["rmse"] / r["cen"] for i in G] for r in rs])
        write(f"S_B{B}", dict(I=G, rel=np.median(rel, 0)))
        v = []
        for r in rs:
            bss, A, _ = S.geometry(B, r["seed"]); kap = rho_bar(metropolis(A), 0.0)
            E0 = r["m12_I1"]["dev0"]; V = np.mean([r[f"m12_I{i}"]["V"] for i in G]); lam = r["lam"]
            v.append(np.log((1 - lam) * E0 / (kap * V)) / np.log(1 / lam))
        ie["B"].append(B); ie["I"].append(np.median(v))
    write("S_ieps", ie)
    for B, i in zip(ie["B"], ie["I"]):
        write(f"S_ieps_B{B}", dict(I=[i], y=[0.875]))      # marker row below the curves


def data_U():
    d = load("U"); I = np.array(d["I_grid"])
    for mdl in ("iid", "burst", "oneway", "priors"):
        a = d["agg"][mdl]; cen = np.array(a["cen"])                   # paired per-seed ratios
        write(f"U_{mdl}", dict(I=I, rel=[np.median(np.array(a[f"I{i}"]["rmse"]) / cen) for i in I]))


def data_R():
    d = load("R")
    for j, nm in (("0", "rsu"), ("1", "ego")):
        m = d["model"][j]; rows = np.array(m["bins"])
        write(f"R_az_{nm}", dict(r=rows[:, 0], sd=np.degrees(np.sqrt(rows[:, 1]))))
        rr = np.linspace(8, 140, 60)
        write(f"R_azfit_{nm}", dict(r=rr, sd=np.degrees(np.sqrt(m["sigma_c"] ** 2 / rr ** 2 + m["sigma_0"] ** 2))))
    sub = [r for r in d["records"] if r["det_any"] >= d["det_min"]]
    for k in ("rsu", "ego", "cen", "dist8", "cen_gauss"):
        v = np.sort([r[k] for r in sub])
        write(f"R_cdf_{k}", dict(rmse=v, cdf=np.arange(1, len(v) + 1) / len(v)))


if __name__ == "__main__":
    for f in (data_B, data_D, data_J, data_A, data_G, data_T, data_S, data_U, data_R):
        f()
