"""Generate numbers.tex: every experimental number quoted in the manuscript,
computed from results/exp_*.json. Never type these numbers by hand."""
import json
import numpy as np
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent.parent / "numbers.tex"
M = {}


def load(n):
    return json.load(open(RES / f"exp_{n}.json"))


def f(x, nd=2):
    return f"{x:.{nd}f}"


def g(x, sig=2):
    """significant-figure formatting for table entries"""
    if not np.isfinite(x) or x > 1e3:
        return "div."
    return f"{x:#.{sig}g}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")   # keep trailing zeros


def pct(x):
    return f"{100 * x:.0f}" if abs(x) >= 0.095 else f"{100 * x:.1f}"


# ------------------------------------------------------------------ Exp B
def exp_B():
    d = load("B"); I = np.array(d["I_grid"]); ag = d["agg"]; meta = d["meta"]
    lam = meta["ring"]["lam"]["0.0"]; kap = meta["ring"]["rho"]["0.0"]; B = d["B"]
    key = lambda topo, m, p=0.0, rule="compensating", fus="hcmci": f"{topo}_b{m}_p{p}_{rule}_{fus}"
    mean = lambda a, i, k: float(np.mean(a[f"I{i}"][k]))
    M["BsigRing"] = f(meta["ring"]["sigma2"]); M["BsigLine"] = f(meta["line"]["sigma2"])
    M["BlamA"], M["BlamB"], M["BlamC"] = (f(meta["ring"]["lam"][k]) for k in ("0.0", "0.2", "0.4"))
    M["Bkappa"] = f(kap)
    rat = []
    for k in (key("ring", None), key("line", None)):
        a = ag[k]
        act = np.array([mean(a, i, "act") for i in I]); bnd = np.array([mean(a, i, "bnd") for i in I])
        ok = act > 1e-18 * act[0]
        rat += list(bnd[ok] / act[ok])
    M["BfacLo"], M["BfacHi"] = f(min(rat), 1), f(max(rat), 1)
    # Theorem 2 check: realised / bound per time step, averaged over steps and seeds
    mx = max(mean(a, i, "ratio") for a in ag.values() for i in I if a["bits"] is not None)
    M["BmaxRatio"] = f(mx, 2)
    names = {4: "A", 6: "B", 8: "C"}
    ith, ieps = {}, {}
    for m, L in names.items():
        ap, ac = ag[key("ring", m, rule="plain")], ag[key("ring", m)]
        actp = np.array([mean(ap, i, "act") for i in I]); actc = np.array([mean(ac, i, "act") for i in I])
        E0 = np.mean(ap["I1"]["dev0"]); V = np.mean([mean(ap, i, "V") for i in I])
        ith[m] = np.log(B * E0 * np.log(1 / lam) / V) / np.log(1 / lam)
        M[f"BIth{L}"] = f"{int(round(ith[m]))}"
        M[f"BImin{L}"] = f"{int(I[int(np.argmin(actp[:40]))])}"
        E0c = np.mean(ac["I1"]["dev0"]); Vc = np.mean([mean(ac, i, "V") for i in I])
        ieps[m] = np.log((1 - lam) * E0c / (kap * Vc)) / np.log(1 / lam)      # eps = 1
        M[f"BIeps{L}"] = f"{int(round(ieps[m]))}"
        floor = np.mean(actc[I >= 40])
        M[f"BIfloor{L}"] = f"{int(I[int(np.argmax(actc <= 2 * floor))])}"
        M[f"BfloorGain{L}"] = f(np.min(actp) / floor, 1)
    M["BperBit"] = f((ith[8] - ith[6]) / 2, 1)          # m = 4: argument of (Istar) < 1
    M["BperBitEps"] = f((ieps[8] - ieps[4]) / 4, 1)
    M["BperBitTh"] = f(2 * np.log(2) / np.log(1 / lam), 1)
    cen = np.mean(ag[key("ring", 6)]["cen"])
    gaps = [abs(mean(ag[key("ring", m)], i, "rmse") / cen - 1) for m in (6, 8) for i in I if i >= 5]
    M["BrmseGap"] = f"{100 * max(gaps):.0f}" if max(gaps) >= 0.01 else "1"
    ap4, ac4 = ag[key("ring", 4, rule="plain")], ag[key("ring", 4)]
    big = [i for i in I if i >= 21]
    div = [r > 3 * cen for i in big for r in ap4[f"I{i}"]["rmse"]]
    M["BplainDiv"] = f"{100 * np.mean(div):.0f}"
    M["BcompFourGap"] = f"{100 * max(abs(mean(ac4, i, 'rmse') / cen - 1) for i in I if i >= 5):.1f}"
    M["BcenRMSE"] = f(cen, 3)
    hc, cm = ag[key("ring", None)], ag[key("ring", None, fus="cm")]
    av = ag[key("ring", None, fus="hcmci_avg")]
    for nm, a in (("Hc", hc), ("Cm", cm), ("Avg", av)):
        M[f"B{nm}One"] = f(mean(a, 1, "rmse") / cen)
        M[f"B{nm}Two"] = f(mean(a, 2, "rmse") / cen)
        M[f"B{nm}NeesOne"] = f(mean(a, 1, "nees"), 1)
        M[f"B{nm}CovOne"] = f"{100 * mean(a, 1, 'cov'):.0f}"
    M["BcenNees"] = f(np.mean(hc["cen_nees"]), 1)
    M["BcenCov"] = f"{100 * np.mean(hc['cen_cov']):.0f}"
    return ith


# ------------------------------------------------------------------ Exp D
def exp_D():
    d = load("D"); bits = d["bits"]; ag = d["agg"]
    # Theorem 3 needs the margin omega < 1; use the REALISED margin along the actual
    # quantized filters (not the centralized-reference proxy gamma)
    om = lambda key, b: np.max(ag[key][f"b{b}"]["omega_max"])
    mmin = min(b for b in bits if om("anchored_0", b) < 1)
    facs = []
    for key in ("anchored_0", "absolute_0", "absolute_1000", "absolute_10000"):
        a = ag[key]
        for b in bits:
            m = np.array(a[f"b{b}"]["dev"]); mse = np.array(a[f"b{b}"]["mse"])
            # same set as the plotted markers: no failed seed, realised margin < 1 everywhere
            if np.all(np.isfinite(m)) and np.all(mse <= 1e2) and om(key, b) < 1:
                r = np.mean(a[f"b{b}"]["pred_exc"]) / np.mean(m)
                facs.append(max(r, 1 / r))
    M["DmMin"] = str(mmin)
    M["DfacHi"] = f(max(facs), 1)
    M["DomThree"] = f(om("anchored_0", 3), 1)
    M["DomLast"] = f(om("anchored_0", mmin - 1), 1)
    M["DmLast"] = str(mmin - 1)
    # absolute frame at 10 km: realised margin and SPD fallbacks where it fails
    divb = [b for b in bits if np.all(np.array(ag["absolute_10000"][f"b{b}"]["mse"]) > 1e2)]
    top = max(divb)                                           # largest bit width that still diverges
    M["DabsTop"] = str(top)
    M["DabsOmTop"] = f(om("absolute_10000", top), 2)
    M["DabsSpdTop"] = str(int(np.sum(ag["absolute_10000"][f"b{top}"]["spdfail"])))
    lost = [b for b in bits if np.sum(ag["absolute_10000"][f"b{b}"]["spdfail"]) > 0]
    M["DabsLostMax"] = str(max(lost)) if lost else "--"     # SPD safeguard triggered
    viol = [b for b in bits if om("absolute_10000", b) >= 1]
    M["DabsOmLostMax"] = str(max(viol)) if viol else "--"    # sufficient margin violated
    M["DnPlot"] = str(sum(1 for key in ("anchored_0", "absolute_0", "absolute_1000", "absolute_10000")
                          for b in bits if np.all(np.isfinite(ag[key][f"b{b}"]["dev"]))
                          and np.all(np.array(ag[key][f"b{b}"]["mse"]) <= 1e2) and om(key, b) < 1))

    def need(key):
        a = ag[key]; thr = 0.01 * np.mean(a["cen_mse"])
        for b in bits:
            m = np.array(a[f"b{b}"]["dev"])
            if np.all(np.isfinite(m)) and np.mean(m) < thr:
                return b
        return None
    M["DanchBits"] = str(need("anchored_0"))
    M["DabsA"], M["DabsB"], M["DabsC"] = (str(need(k)) for k in ("absolute_0", "absolute_1000", "absolute_10000"))

    def alldiv(key):
        a = ag[key]; last = None
        for b in bits:
            m = np.array(a[f"b{b}"]["mse"])
            if np.all(~np.isfinite(m) | (m > 1e2)):
                last = b
        return str(last)
    M["DdivB"], M["DdivC"] = alldiv("absolute_1000"), alldiv("absolute_10000")


# ------------------------------------------------------------------ Exp J
def exp_J():
    d = load("J"); r = d["rows"]; m0 = d["mse0"]
    dyn, sta, rnd, jnt = [], [], [], []
    for p in d["rhos"][1:]:
        wc = r[f"naive_wc_{p}"][0]
        dyn.append(abs((m0 + p * p * d["lam_dyn"]) / wc - 1))
        sta.append(1 - (m0 + p * p * d["lam_static"]) / wc)
        rnd.append(wc / r[f"naive_rand_{p}"][0]); jnt.append(wc / r[f"joint_wc_{p}"][0])
    M["JdynErr"] = f"{100 * max(dyn):.0f}" if max(dyn) >= 0.01 else "1"
    M["JstaticLo"], M["JstaticHi"] = f"{100 * min(sta):.0f}", f"{100 * max(sta):.0f}"
    M["JrandLo"], M["JrandHi"] = f(min(rnd), 1), f(max(rnd), 1)
    M["JjointLo"], M["JjointHi"] = f(min(jnt), 1), f(max(jnt), 1)
    M["JexWc"] = f(r["naive_wc_1.0"][0], 3); M["JexPred"] = f(m0 + d["lam_dyn"], 3)


# ------------------------------------------------------------------ Exp A
def exp_A():
    d = load("A"); allruns = d["runs"]
    runs = [r for r in allruns if "fus" not in r]           # Part A2 (centralised EKF)
    pts = {}
    for r in runs:
        e = np.array(r["err2"])
        pts.setdefault((r["method"], r["eta"]), {})[r["seed"]] = dict(
            rate=np.mean(r["rate"]), rmse=np.sqrt(e[10:].mean()),
            mse1=e[10:, 0].mean(), mse2=e[10:, 1].mean())
    etas = sorted({k[1] for k in pts if k[0] == "mpcA"})
    for crit in ("A", "D"):
        gains, share = [], []
        for eta in etas:
            a, b = pts[(f"mpc{crit}", eta)], pts[(f"myopic{crit}", eta)]
            s = sorted(set(a) & set(b))
            ra = np.mean([a[k]["rmse"] for k in s]); rb = np.mean([b[k]["rmse"] for k in s])
            gains.append(1 - ra / rb)
            d1 = np.mean([b[k]["mse1"] - a[k]["mse1"] for k in s])
            d2 = np.mean([b[k]["mse2"] - a[k]["mse2"] for k in s])
            share.append(d1 / (d1 + d2) if d1 + d2 > 0 else np.nan)
        M[f"Again{crit}lo"], M[f"Again{crit}hi"] = pct(min(gains)), pct(max(gains))
        M[f"Ashare{crit}"] = f"{100 * np.nanmean(share):.0f}"
    drops = []
    for eta in etas:
        for crit in ("A", "D"):
            a, b = pts[(f"mpc{crit}", eta)], pts[(f"myopic{crit}", eta)]
            drops.append((eta, np.mean([v["rate"] for v in b.values()]) - np.mean([v["rate"] for v in a.values()])))
    M["ArateMax"] = f(max(x for _, x in drops), 1)
    M["ArateEta"] = f"{min(e for e in etas if all(x < 0.5 for ee, x in drops if ee >= e))}"

    def frontier(meth, rate):
        xs, ys = [], []
        for eta in etas:
            v = pts[(meth, eta)]
            xs.append(np.mean([u["rate"] for u in v.values()])); ys.append(np.mean([u["rmse"] for u in v.values()]))
        o = np.argsort(xs)
        return float(np.interp(rate, np.array(xs)[o], np.array(ys)[o], left=np.nan, right=np.nan))
    hi = 46.0; mid = 44.0
    M["ArateHi"], M["ArateMid"] = f"{hi:.0f}", f"{mid:.0f}"
    M["AfrMpcA"] = f(frontier("mpcA", hi), 3); M["AfrMyoA"] = f(frontier("myopicA", hi), 3)
    M["AfrMpcD"] = f(frontier("mpcD", hi), 3); M["AfrMyoD"] = f(frontier("myopicD", hi), 3)
    M["AfrSnapMid"] = f(frontier("snapshot", mid), 3)
    M["AfrSnapAMid"] = f(frontier("snapshotA", mid), 3)
    M["AfrSnapAHi"] = f(frontier("snapshotA", hi), 3)
    # paired MPC-vs-myopic RMSE reduction per eta (A-criterion), 95% t-interval over seeds
    from scipy.stats import t as tdist
    sig = 0
    for eta in etas:
        a_, b_ = pts[("mpcA", eta)], pts[("myopicA", eta)]
        ks = sorted(set(a_) & set(b_))
        dlt = np.array([1 - a_[k]["rmse"] / b_[k]["rmse"] for k in ks])
        lo = dlt.mean() - tdist.ppf(0.975, len(dlt) - 1) * dlt.std(ddof=1) / np.sqrt(len(dlt))
        sig += lo > 0
    M["AsigA"], M["Anetas"] = str(sig), str(len(etas))
    # Part A3: integrated distributed loop, paired with the centralised run (same seed, eta)
    cenr = {(r["eta"], r["seed"]): np.sqrt(np.mean(np.array(r["err2"])[10:])) for r in runs if r["method"] == "mpcA"}
    integ = [r for r in allruns if "fus" in r]
    if integ:
        def ratio(method, name):
            v = [np.sqrt(np.mean(np.array(r["err2"])[10:])) / cenr[(r["eta"], r["seed"])]
                 for r in integ if r["method"] == method and r["fus"]["name"] == name]
            return np.median(v), np.percentile(v, 90)
        for name, nm in (("full_I2", "Full"), ("m8_I2", "Eight"), ("m8_I2_p02", "EightLoss"),
                         ("m4_I2_p02", "FourLoss"), ("m8_I1", "EightOne")):
            med, p90 = ratio("mpcA", name)
            M[f"AInt{nm}"], M[f"AInt{nm}P"] = f(med, 3), f(p90, 2)
        # safeguard (SPD) rejections, MPC-A only; unit: node-target update, 780 per run
        def spd_rate(name):
            rr = [r for r in integ if r["method"] == "mpcA" and r["fus"]["name"] == name]
            n = sum(r.get("spdfail", 0) for r in rr); den = 780 * len(rr)
            return n, den
        n4, den = spd_rate("m4_I2_p02"); n8, _ = spd_rate("m8_I2_p02")
        M["AIntSpdFour"] = f"{n4:,}".replace(",", "{,}"); M["AIntSpdEight"] = str(n8)
        M["AIntSpdDen"] = f"{den:,}".replace(",", "{,}")
        # tail table over the 40 (seed, eta) rows of each integrated MPC-A configuration
        rows = []
        for name, lab in (("full_I2", "full, $I{=}2$"), ("m8_I2", "8 b, $I{=}2$"),
                          ("m8_I2_p02", "8 b, $I{=}2$, $p{=}0.2$"), ("m8_I1", "8 b, $I{=}1$"),
                          ("m4_I2_p02", "4 b, $I{=}2$, $p{=}0.2$")):
            v = np.array([np.sqrt(np.mean(np.array(r["err2"])[10:])) / cenr[(r["eta"], r["seed"])]
                          for r in integ if r["method"] == "mpcA" and r["fus"]["name"] == name])
            n, dn = spd_rate(name)
            mx = np.max(v); mxs = f"{mx:.3g}" if mx < 100 else f"{mx:,.0f}".replace(",", "{,}")
            rows.append(f"{lab} & {np.median(v):.3f} & {np.percentile(v, 90):.2f} & {mxs} & "
                        f"{int(np.sum(v > 3))}/{len(v)} & {100 * n / dn:.2f}\\\\")
            if name == "m8_I1":
                M["AIntEightOneMax"] = f"{mx:.0f}"; M["AIntEightOneFail"] = str(int(np.sum(v > 3)))
            if name == "m8_I2_p02":
                M["AIntEightLossFail"] = str(int(np.sum(v > 3)))
            if name == "m4_I2_p02":
                M["AIntFourLossFail"] = str(int(np.sum(v > 3)))
        M["TabARows"] = "\n".join(rows)
        # MPC-A vs myopic-A relative gain with 8 bits, I=2, p=0.2: average the two eta values
        # within each seed (shared trajectories/channels/noise), t-interval over the seeds
        gs = []
        for sd in sorted({r["seed"] for r in integ}):
            ge = []
            for eta in (0.2, 0.6):
                rr = {r["method"]: np.sqrt(np.mean(np.array(r["err2"])[10:])) for r in integ
                      if r["eta"] == eta and r["seed"] == sd and r["fus"]["name"] == "m8_I2_p02"}
                if len(rr) == 2:
                    ge.append(1 - rr["mpcA"] / rr["myopicA"])
            if len(ge) == 2:
                gs.append(np.mean(ge))
        gs = np.array(gs); hw = tdist.ppf(0.975, len(gs) - 1) * gs.std(ddof=1) / np.sqrt(len(gs))
        M["AIntGain"] = pct(gs.mean())
        M["AIntGainLo"], M["AIntGainHi"] = pct(gs.mean() - hw), pct(gs.mean() + hw)
        M["AIntNseed"] = str(len(gs))
    bay = [frontier(m, mid) for m in ("mpcA", "myopicA", "mpcD", "myopicD")]
    bay = [x for x in bay if np.isfinite(x)]
    M["AfrBayesLo"], M["AfrBayesHi"] = f(min(bay), 3), f(max(bay), 3)


# ------------------------------------------------------------------ Exp G
def exp_G():
    d = load("G"); R = d["records"]
    v = lambda k: np.array([r[k] for r in R])
    cen, rsu, ego, d8, d6 = v("cen"), v("rsu"), v("ego"), v("dist8"), v("dist6")
    best = np.minimum(rsu, ego)
    M["Gn"] = f"{len(R)}"
    M["Gcons"], M["Gconsp"] = f(d["consistency"]["median"]), f(d["consistency"]["p90"])
    if "frames" in d:
        M["Gframes"] = f"{d['frames']['total']:,}".replace(",", "{,}")
        M["Gseclo"], M["Gsechi"] = f"{d['frames']['min'] / 10:.0f}", f"{d['frames']['max'] / 10:.0f}"
        M["Gidjump"] = f(d["id_jump"]["median"])
    M["Grsu"] = f"{100 * np.mean(v('frac_both') + v('frac_rsu_only')):.0f}"
    M["Gego"] = f"{100 * np.mean(v('frac_both') + v('frac_ego_only')):.0f}"
    M["Gboth"] = f"{100 * np.mean(v('frac_both')):.0f}"
    th = [x for r in R for x in r["thm1"]]
    k1 = [x for x in th if x["K"] == 1]; k5 = [x for x in th if x["K"] == 5 and x["lam_c"] > 0]
    M["Gsnap"] = f"{len(k1):,}".replace(",", "{,}")
    t = np.array([x["bound"] / x["lam_c"] for x in k1])
    M["Gtight"], M["Gtightlo"], M["Gtighthi"] = f(np.median(t)), f(np.percentile(t, 10)), f(np.percentile(t, 90))
    gn = np.array([x["lam_c"] / max(x["lam_r"], x["lam_e"]) for x in k5 if max(x["lam_r"], x["lam_e"]) > 0])
    M["Ggain"], M["Ggainlo"] = f(np.median(gn), 1), f(np.percentile(gn, 10), 1)
    M["Gbetter"] = f"{100 * np.mean(cen < best * 0.999):.0f}"
    M["Gratio"] = f(np.percentile(best / cen, 90))
    worse = cen > best * 1.001
    M["Gworse"] = f"{100 * np.mean(worse):.1f}"
    M["Gworsemax"] = f"{100 * (np.max(cen / best) - 1):.0f}"
    M["Gdeight"] = f"{100 * (np.median(d8 / cen) - 1):.1f}"
    M["Gdeightp"] = f"{100 * (np.percentile(d8 / cen, 95) - 1):.0f}"
    M["Gdsix"] = f"{100 * np.mean(d6 > 10 * cen):.0f}"


# ------------------------------------------------------------------ Exp I and E (table)
def exp_IE():
    I = load("I"); info = I["info"]; ag = I["agg"]
    med = {k: np.array([np.median(a[k]) for a in ag.values()]) for k in next(iter(ag.values())) if k != "spd"}
    sp = [i["mean_speed"] for i in info.values()]; acc = [i["p95_acc"] for i in info.values()]
    M["Ispeedlo"], M["Ispeedhi"] = f(min(sp), 1), f(max(sp), 1)
    M["Iacclo"], M["Iacchi"] = f(min(acc), 1), f(max(acc), 1)
    M["Istop"] = f"{100 * max(i['frac_stopped'] for i in info.values()):.0f}"
    pc = np.array([i["pcrb_rmse"] for i in info.values()])
    M["IpcrbLo"], M["IpcrbHi"] = f(np.min(med["cen"] / pc)), f(np.max(med["cen"] / pc))
    rel = lambda k: med[k] / med["cen"] - 1
    M["IfullMax"] = f(100 * np.max(rel("I8")), 1)
    M["ItenMax"] = f(100 * np.max(rel("I8_b10")), 1)
    M["IeightMax"] = f"{100 * np.max(rel('I8_b8')):.0f}"
    M["IfailMax"] = f(np.max(med["I8_b8_p02"] - med["cen"]), 3)
    qk = ("I8_b8", "I8_b10", "I8_b12", "I8_b16", "I8_b8_p02")
    M["Iquant"] = f(100 * max(np.max(np.abs(rel(k))) for k in qk), 1)
    M["ItwoMed"] = f"{100 * np.median(rel('I2')):.0f}"
    M["IabsFac"] = f(np.median(med["I8_b8_abs"] / med["I8_b8"]), 1)
    M["IabsWorst"] = f(np.max(med["I8_b8_abs"]), 1)
    M["IabsSixteen"] = f(np.median(med["I8_b16_abs"]), 3)
    from isac_core import metropolis, line_adj, lambda_bar
    M["Ilam"] = f(lambda_bar(metropolis(line_adj(4)), 0.0))
    rows = [r"Boreas$^\dagger$ & " + " & ".join(g(np.median(med[k])) if k else "--" for k in
            ("cen", None, "I8", "I8_b8", "I8_b12", "I8_b16", "I8_b12_abs")) + r"\\"]
    E = load("E"); names = {"city_0_newyork_3p5_s": "NYC 3.5", "city_1_losangeles_3p5": "LA 3.5",
                            "city_4_phoenix_3p5": "Phoenix 3.5", "city_0_newyork_28": "NYC 28"}
    em = {n: {k: float(np.median(v)) for k, v in a.items() if k != "spd"} for n, a in E["agg"].items()}
    for n, a in em.items():
        rows.append(f"{names[n]} & " + " & ".join(g(a[k]) for k in
                    ("cen", "best_single", "I5", "I5_b8", "I5_b12", "I5_b16", "I5_b12_abs")) + r"\\")
    M["TabRows"] = "\n".join(rows)
    M["TabDivNote"] = r" div.: diverged ($>10^3$~m)." if any("div." in r for r in rows) else ""
    # paired per-trajectory ratios (trajectory errors are bimodal: some lose
    # track in coverage holes), relative to full-precision distributed fusion
    pr = lambda a, k, ref="I5": float(np.median(np.array(a[k]) / np.array(a[ref])))
    need = []
    for n, a in E["agg"].items():
        nb = next((b for b in (8, 10, 12, 14, 16) if abs(pr(a, f"I5_b{b}") - 1) <= 0.1), None)
        need.append(nb if nb else 99)
    M["Ebits"] = str(min(need)) if min(need) == max(need) else f"{min(need)}--{max(need)}"
    v16 = 100 * max(abs(pr(a, "I5_b16") - 1) for a in E["agg"].values())
    M["EsixMax"] = "1" if v16 < 1 else f"{v16:.0f}"
    ab = [pr(a, "I5_b12_abs", "I5_b12") for a in E["agg"].values()]
    M["EabsTwelveLo"], M["EabsTwelveHi"] = f(min(ab), 1), f(max(ab), 1)
    sg = [pr(a, "best_single", "cen") for a in E["agg"].values()]
    M["EsingleLo"], M["EsingleHi"] = f(min(sg), 1), f(max(sg), 1)
    e8 = [pr(a, "I5_b8") for a in E["agg"].values()]
    M["EeightFail"] = str(sum(r > 1.5 for r in e8)); M["Escenes"] = str(len(e8))
    M["EeightLo"], M["EeightHi"] = f(min(e8), 1), (f(max(e8), 1) if max(e8) < 10 else f"{max(e8):.0f}")
    M["EphxMulti"] = f"{100 * em['city_4_phoenix_3p5']['cov_multi']:.0f}"
    # SPD-safeguard rejections with 8 bits: node updates (T=300 steps x 3 BSs per trajectory)
    import exp_E_deepmimo as EE
    rates = [100 * sum(sp.get("I5_b8", 0) for sp in a["spd"]) / (EE.T * 3 * len(a["spd"]))
             for a in E["agg"].values()]
    M["ESpdLo"], M["ESpdHi"] = f(min(rates), 1), f(max(rates), 1)
    M["ESpdTwelve"] = str(sum(sp.get("I5_b12", 0) for a in E["agg"].values() for sp in a["spd"]))


# ------------------------------------------------------------------ Exp R (recorded radar)
def exp_R():
    d = load("R"); R = d["records"]; mod = d["model"]; hr = d["held_resid"]
    M["RsigcR"], M["RsigcE"] = f(mod["0"]["sigma_c"]), f(mod["1"]["sigma_c"])
    M["RsigrR"], M["RsigrE"] = f(mod["0"]["sigma_r"]), f(mod["1"]["sigma_r"])
    M["RsigrrR"], M["RsigrrE"] = f(mod["0"]["sigma_rr"]), f(mod["1"]["sigma_rr"])
    M["RbiasR"], M["RbiasE"] = f(mod["0"]["bias"][1]), f(mod["1"]["bias"][1])
    M["RhbiasR"], M["RhbiasE"] = f(hr["0"]["bias"][1]), f(hr["1"]["bias"][1])
    M["Rkappa"] = f"{d['kappa']:g}"
    M["Rn"] = str(len(R))
    M["RdetAny"] = f"{100 * np.median([r['det_any'] for r in R]):.0f}"
    S = [r for r in R if r["det_any"] >= d["det_min"]]
    M["Rnsub"] = str(len(S))
    v = lambda k, rs=S: np.array([r[k] for r in rs])
    for k, nm in (("rsu", "Rrsu"), ("ego", "Rego"), ("cen", "Rcen"), ("cen_gauss", "Rgauss")):
        M[nm] = f(np.median(v(k)))
    cen = v("cen")
    M["Rbetter"] = f"{100 * np.mean(cen < np.minimum(v('rsu'), v('ego')) * 0.999):.0f}"
    M["RgaussFac"] = f(np.median(cen) / np.median(v("cen_gauss")), 1)
    for m in (6, 8, 10, 12, 16):
        gap = np.median(v(f"dist{m}") / cen) - 1
        if abs(gap) <= 0.01:
            M["RdistBits"] = str(m); M["RdistRatio"] = f(1 + gap, 3)
            break
    tg = np.array(d["tune_grid"])                    # (sigma_a, kappa, train median RMSE)
    flat = tg[(tg[:, 0] >= 3)]
    M["RtuneLo"], M["RtuneHi"] = f(flat[:, 2].min()), f(flat[:, 2].max())
    M["Rntrain"] = str(d["n_train_tracks"])
    M["RabsEight"] = f(100 * (np.median(v("abs8")) / np.median(cen) - 1), 0)
    rat = v("cen_uncal") / cen                       # paired per-track effect of skipping calibration
    M["RcalRatio"] = f(np.median(rat), 3); M["RcalFrac"] = f"{100 * np.mean(rat > 1):.0f}"
    inc = v("cen_uncal_mse") - v("cen_mse"); pred = v("prop1_pred_bias2")
    M["RpropFac"] = f(np.median(pred) / np.median(inc), 1) if np.median(inc) > 0 else "--"
    M["RpropCorr"] = f(np.corrcoef(pred, inc)[0, 1])
    # all held-out tracks (no detection-based selection)
    va = lambda k: np.array([r[k] for r in R])
    M["RcenAll"], M["RcenAllP"] = f(np.median(va("cen"))), f(np.percentile(va("cen"), 90))
    M["RcenP"] = f(np.percentile(cen, 90))
    M["RgaussAll"] = f(np.median(va("cen_gauss")))
    M["RdistAll"] = f(np.median(va(f"dist{M['RdistBits']}") / va("cen")), 3)
    # scene-cluster bootstrap (95%) of the median paired ratio dist/cen and of cen/best-single
    rng = np.random.default_rng(0)
    scenes = sorted({r["scene"] for r in S})
    by = {sc: [r for r in S if r["scene"] == sc] for sc in scenes}
    def boot(fn):
        vals = []
        for _ in range(2000):
            pick = [r for sc in rng.choice(scenes, len(scenes)) for r in by[sc]]
            vals.append(fn(pick))
        return np.percentile(vals, [2.5, 97.5])
    m_ = M["RdistBits"]
    lo, hi = boot(lambda rs: np.median([r[f"dist{m_}"] / r["cen"] for r in rs]))
    M["RdistLo"], M["RdistHi"] = f(lo, 3), f(hi, 3)
    lo, hi = boot(lambda rs: np.median([min(r["rsu"], r["ego"]) / r["cen"] for r in rs]))
    M["RcoopRatio"] = f(np.median(np.minimum(v("rsu"), v("ego")) / cen), 2)
    M["RcoopLo"], M["RcoopHi"] = f(lo, 2), f(hi, 2)
    M["Rscenes"] = str(len(scenes))
    best = np.minimum(v("rsu"), v("ego"))
    row = r"V2X-Radar$^\ddagger$ & " + " & ".join(g(np.median(x)) for x in
          (cen, best, v("distfull"), v("dist8"), v("dist12"), v("dist16"), v("abs12"))) + r"\\"
    M["TabRowR"] = row


# ------------------------------------------------------------------ Exp U (robustness)
def exp_U():
    d = load("U"); ag = d["agg"]; I = d["I_grid"]
    rel = lambda mdl, i: np.median(np.array(ag[mdl][f"I{i}"]["rmse"]) / np.array(ag[mdl]["cen"]))  # paired per seed
    rat = lambda mdl, i: np.mean(ag[mdl][f"I{i}"]["act"]) / np.mean(ag[mdl][f"I{i}"]["bnd"])
    big = [i for i in I if i >= 2]
    late = [i for i in I if i >= 8]                  # converged regime
    for mdl, nm in (("iid", "Iid"), ("burst", "Burst"), ("oneway", "Oneway"), ("priors", "Priors")):
        gp = 100 * max(abs(rel(mdl, i) - 1) for i in (big if mdl == "priors" else late))
        M[f"U{nm}Gap"] = f(gp, 2) if gp < 0.1 else f(gp, 1)
        M[f"U{nm}GapTwo"] = f(100 * (rel(mdl, 2) - 1), 1)
        M[f"U{nm}Ratio"] = f(max(rat(mdl, i) for i in I), 2 if mdl == "iid" else 0)
    M["UPriorsOne"] = f(100 * (rel("priors", 1) - 1), 1)
    M["Ubursts"] = f"{d['burst']:g}"


# ------------------------------------------------------------------ Exp T (flooding comparison)
def exp_T():
    d = load("T"); S = d["summary"]
    TOL = 0.05                                       # at most 5 % worse than centralized
    ok = lambda q: q["rel"] <= 1 + TOL and q["frac_div"] == 0

    def cheapest(key, fam, bk="bits8"):
        pts = [q for q in S[key]["points"] if q["family"] == fam and ok(q)]
        return min(pts, key=lambda q: q[bk]) if pts else None

    def fmt(x):
        return f"{x:,.0f}".replace(",", "{,}")
    for key, nm in (("ring6_p0.0", "RingSix"), ("ring16_p0.0", "RingSixteen"), ("circ16_p0.0", "CircSixteen"),
                    ("rr64_p0.0", "RrSixtyFour"), ("rr128_p0.0", "RrOneTwentyEight"),
                    ("rr128_p0.2", "RrLoss")):
        fl = cheapest(key, "floodarq" if key.endswith("0.2") else "flood", "bits32")
        co = cheapest(key, "cons")
        M[f"Tfl{nm}"] = fmt(fl["bits32"]) if fl else "--"
        M[f"Tco{nm}"] = fmt(co["bits8"]) if co else "--"
        M[f"Tlatfl{nm}"] = f(fl["latency"], 0 if fl["latency"] == int(fl["latency"]) else 1) if fl else "--"
        M[f"Tlatco{nm}"] = f"{co['latency']:.0f}" if co else "--"
    for key, nm in (("ring16_p0.2", "RingLoss"), ("circ16_p0.2", "CircLoss")):
        fl = [q for q in S[key]["points"] if q["family"] == "flood"]
        M[f"Tflrel{nm}"] = f"{min(q['rel'] for q in fl):.0f}" if min(q["rel"] for q in fl) >= 10 else f(min(q["rel"] for q in fl), 1)
        M[f"Tfldiv{nm}"] = f"{100 * max(q['frac_div'] for q in fl):.0f}"
        co = [q for q in S[key]["points"] if q["family"] == "cons" and q["frac_div"] == 0]
        M[f"Tcorel{nm}"] = f(min(q["rel"] for q in co), 2)
    M["Ttol"] = f"{100 * TOL:.0f}"


# ------------------------------------------------------------------ Exp S (network size)
def exp_S():
    import exp_S_network_scale as S
    from isac_core import metropolis, rho_bar
    d = load("S"); G = d["I_grid"]; runs = d["runs"]
    M["SmaxRatio"] = f(max(r[f"m{m}_I{i}"]["ratio"] for r in runs for m in d["bits"] for i in G), 2)
    real, ieps, div8, div12, om8 = {}, {}, [], [], []
    for r in runs:
        bss, A, _ = S.geometry(r["B"], r["seed"])
        kap = rho_bar(metropolis(A), r["p"], n_mc=1000)
        for m in d["bits"]:
            E0 = r[f"m{m}_I1"]["dev0"]; V = np.mean([r[f"m{m}_I{i}"]["V"] for i in G]); lam = r["lam"]
            ie = np.log((1 - lam) * E0 / (kap * V)) / np.log(1 / lam)
            dv = [i for i in G if r[f"m{m}_I{i}"]["rmse"] > 3 * r["cen"]]
            (div8 if m == 8 else div12).append(len(dv) / len(G))
            if m == 8:
                om8 += [r[f"m8_I{i}"]["omega_max"] for i in dv]
            if m == 12:
                ieps.setdefault(r["B"], []).append(ie)
                rr = next((i for i in G if r[f"m12_I{i}"]["rmse"] <= 1.05 * r["cen"]), np.nan)
                real.setdefault(r["B"], []).append(rr)
    Bs = sorted(ieps)
    M["SiepsLo"], M["SiepsHi"] = f"{np.median(ieps[Bs[0]]):.0f}", f"{np.median(ieps[Bs[-1]]):.0f}"
    med = [np.nanmedian(real[b]) for b in Bs]
    M["SrealLo"], M["SrealHi"] = f"{min(med):.0f}", f"{max(med):.0f}"
    M["SrealMax"] = f"{np.nanmax([x for b in Bs for x in real[b]]):.0f}"
    M["SdivEight"] = f(100 * np.mean(div8), 1)
    M["SdivTwelve"] = f(100 * np.mean(div12), 2)
    M["SomegaMin"] = f(min(om8), 1)
    M["SBmax"] = str(Bs[-1])
    deg = {B: [r["deg"] for r in runs if r["B"] == B] for B in Bs}
    M["SdegLo"], M["SdegHi"] = f(np.mean(deg[Bs[0]]), 1), f(np.mean(deg[Bs[-1]]), 1)
    rel64 = np.median([[r[f"m12_I{i}"]["rmse"] / r["cen"] for i in G] for r in runs
                       if r["B"] == Bs[-1] and r["p"] == 0.0], 0)
    M["SminRel"] = f(rel64.min())


# ------------------------------------------------------------------ Exp V (frozen-message checks)
def exp_V():
    d = load("V"); rows = d["rows"]
    M["VN"] = f"{d['N']:,}".replace(",", "{,}")
    M["Vconf"] = str(len(rows))
    M["VmaxRatio"] = f(max(r["ratio"] for r in rows), 2)      # point ratio mean(error)/mean(bound)
    rmax = max(r['resid'] for r in rows); e = int(np.floor(np.log10(rmax)))
    M["Vresid"] = f"{rmax / 10 ** e:.1f}\\cdot10^{{{e}}}"          # e.g. 6.9\cdot10^{-15}
    M["VdNeg"] = str(sum(r["d_hi"] < 0 for r in rows))
    ok = [r for r in rows if "sand_lo_min" in r]
    M["Vsand"] = str(len(ok))
    M["VsandLo"] = f(min(r["sand_lo_min"] for r in ok), 2)
    M["VsandHi"] = f(max(r["sand_hi_max"] for r in ok), 2)
    M["Vcor"] = f(max(r["cor_ratio"] for r in ok), 2)
    M["Valpha"] = f(max(r["alpha_ratio_max"] for r in rows), 2)
    tab = []
    for r in rows:
        if r["t"] == d["times"][0] and r["I"] in (1, 8, 21) and r["rule"] == "compensating":
            sand = (f"{r['sand_lo_min']:.2f} / {r['sand_hi_max']:.2f}" if "sand_lo_min" in r else "--")
            c = r.get("cor_ratio")
            if c is None:
                cor = "--"
            elif c >= 1e-2:
                cor = f"{c:.3f}"
            else:                                    # two significant digits, no "0.000"
                e = int(np.floor(np.log10(c))); cor = f"${c / 10 ** e:.1f}\\cdot10^{{{e}}}$"
            tab.append(f"{r['m']} & {r['I']} & {r['p']:g} & {r['ratio']:.2f} & "
                       f"{r["omega_bar"]:.2f} & {sand} & {cor}\\\\")
    M["TabVRows"] = "\n".join(tab)


def main():
    import sys
    lenient = "--lenient" in sys.argv          # strict by default: never write partial macros
    for fn in (exp_B, exp_D, exp_J, exp_A, exp_G, exp_IE, exp_R, exp_U, exp_T, exp_S, exp_V):
        try:
            fn()
        except Exception as e:
            if not lenient:
                raise
            print("WARNING", fn.__name__, repr(e))
    bad = [k for k, v in M.items() if any(x in str(v) for x in ("nan", "inf"))]
    if bad and not lenient:
        raise ValueError(f"non-finite claimed numbers: {bad}")
    lines = ["% AUTO-GENERATED by code/make_numbers.py from results/exp_*.json -- do not edit"]
    for k, v in sorted(M.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    OUT.write_text("\n".join(lines) + "\n")
    for k, v in sorted(M.items()):
        if k not in ("TabRows", "TabRowR", "TabVRows", "TabARows"):
            print(f"{k:14s} {v}")
    print(M.get("TabRows", ""))


if __name__ == "__main__":
    main()
