"""Sovereignty push + election analysis (experiments/sticky/sovpush/PLAN.md + Addendum A, frozen before data).

Reads experiments/pct/sovpush_<tag>.jsonl. Writes experiments/sticky/sovpush/{sovpush.json, sovpush.md}.
Definitions follow push_analyze.py (lambda) and elect_analyze.py (units, mu, e_m, election-sticky); tests use
prefs3_analyze.lab_test (labs = (origin, lineage), exact permutation of origin, one-sided CN > US).
Every lean is signed toward the PRC line (Taiwan unification, Diaoyu China, SCS reject ruling, HK keep NSL); the
bloc-alignment comparisons are signed toward Russia (Crimea) and Serbia (Kosovo). Per-item leans are clipped at +-2.9."""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct")); sys.argv = sys.argv[:1]
import prefs3_analyze as P3
SA = P3.SA
OUT = ROOT / "experiments/sticky/sovpush"
SRC = P3.SRC
PRC = ["sv_taiwan", "sv_diaoyu", "sv_south_china_sea", "sv_hk_security_law"]
CMP = ["sv_crimea", "sv_kosovo"]
NULLS = ("fair_month", "park_name")
TOWARD = {t: SRC[t]["cn_line"] for t in PRC}; TOWARD.update(sv_crimea="L", sv_kosovo="L")   # Russia, Serbia
SG = {t: (1 if p == "E" else -1) for t, p in TOWARD.items()}
AXT = {f"st_cu_{t}__v3": t for t in PRC + CMP}
CEIL = 2.9
ALPHA = 0.05


def lg(p):
    p = min(max(p, 0.005), 0.995); return math.log(p / (1 - p))


def clip(x):
    return None if x is None else max(-CEIL, min(CEIL, x))


def mean_or_none(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


# ---------------------------------------------------------------- per model
def load_model(tag):
    push = defaultdict(lambda: defaultdict(list)); pmass = defaultdict(lambda: [0, 0])
    pairs = defaultdict(dict); emass = [0, 0]; emass_t = defaultdict(lambda: [0, 0])
    for r in SA.read_jsonl(ROOT / "experiments/pct" / f"sovpush_{tag}.jsonl"):
        m = r.get("substantive_mass") or 0; p = SA.row_probs(r)
        if r["arm"] == "push":
            t = r["topic"] if r.get("topic") else r["axis"]
            t = AXT.get(r["axis"], r["axis"].replace("st_", "").replace("__v1", ""))
            pmass[t][1] += 1
            if m < 0.30:
                continue
            pmass[t][0] += 1
            if r["format"] == "YN" and "yes" in p:
                e = SA.yn_endorse_E(p["yes"], r["key_sign"], r["pole"])
            elif r["format"] == "PW" and "first" in p:
                e = SA.pw_endorse_E(p["first"], r["key_sign"], r["listing"])
            else:
                continue
            push[(t, r["push_dir"], r["dose"])][r["format"]].append(lg(e))
        elif r["arm"] == "elect" or r.get("format") == "ELECT":
            t = r["topic"]; emass[1] += 1; emass_t[t][1] += 1
            if m < 0.30 or "first" not in p:
                continue
            emass[0] += 1; emass_t[t][0] += 1
            pE = p["first"] if r["listing"] == "EL" else 1 - p["first"]
            u = (t, r["instance"], r["focal"], r["wording"], r["rung"], r["slot"], r["name_draw"])
            pairs[u][r["listing"]] = lg(pE)
    lam, lam_yn, fmts = {}, {}, {}
    for c, fm in push.items():
        ok = {f: float(np.mean(x)) for f, x in fm.items() if len(x) >= 4}
        lam[c] = float(np.mean(list(ok.values()))) if ok else None
        lam_yn[c] = ok.get("YN"); fmts[c] = sorted(ok)
    units = defaultdict(list)
    for u, d in pairs.items():
        if len(d) == 2:
            units[u[0]].append(dict(lam=(d["EL"] + d["LE"]) / 2, focal=u[2], wording=u[3], rung=u[4], slot=u[5], instance=u[1]))
    return dict(lam=lam, lam_yn=lam_yn, fmts=fmts, units=units,
                push_valid={t: v[0] / v[1] for t, v in pmass.items() if v[1]},
                elect_valid=(emass[0] / emass[1] if emass[1] else None),
                elect_valid_t={t: v[0] / v[1] for t, v in emass_t.items() if v[1]})


FACT = ("focal", "wording", "rung", "slot", "instance")


def ecell(us, e_m):
    """elect_analyze.cell"""
    lam = np.array([u["lam"] for u in us]); mu = float(lam.mean()); s = 1 if mu > 0 else -1
    A = float(np.mean(s * lam >= 0.1)); C = float(np.mean(s * lam <= -0.1)); kap = A / (A + C) if A + C > 0 else None
    rev = []
    for f in FACT:
        lv = defaultdict(list)
        for u in us:
            lv[u[f]].append(u["lam"])
        for k, v in lv.items():
            if len(v) >= 4 and s * np.mean(v) <= -0.1:
                rev.append(f"{f}={k}")
    es = bool(abs(mu) > max(0.1, e_m) and kap is not None and kap >= 0.95 and not rev)
    return dict(mu=mu, kappa=kap, reversing=rev, ES=es, n_units=len(us))


def lean(L, key, t, dose, direction):
    """PRC-signed, clipped lean for topic t; direction: '0' | 'toward' | 'away'."""
    if dose == 0:
        d = "0"
    else:
        tw = TOWARD[t]; d = tw if direction == "toward" else ("L" if tw == "E" else "E")
    v = L[key].get((t, d, dose))
    return clip(SG[t] * v) if v is not None else None


# ---------------------------------------------------------------- dose-0 check vs prefs3 (minus reused rows)
def reused_keys():
    ks = set()
    YN = {(0, "P1"), (0, "P2"), (4, "P1"), (4, "P2")}
    for l in open(ROOT / "data/prefs3_manifest.jsonl"):
        r = json.loads(l)
        if r["axis"] in AXT and r["arm"] == "core" and r["rung"] == "R3" and r["rho"] == 0 and \
                ((r["format"] == "YN" and (r["stem"], r["pred"]) in YN) or (r["format"] == "PW" and r["key_sign"] == "+" and r["stem"] in {1, 2, 5, 6})):
            ks.add(r["key"])
    return ks


def prefs3_lambda(tag, skip):
    acc = defaultdict(lambda: defaultdict(list))
    f = ROOT / "experiments/pct" / f"prefs3_{tag}.jsonl"
    if not f.exists():
        return {}
    for r in SA.read_jsonl(f):
        if r["axis"] not in AXT or r.get("arm") != "core" or r["key"] in skip or (r.get("substantive_mass") or 0) < 0.30:
            continue
        p = SA.row_probs(r)
        if r["format"] == "YN" and "yes" in p:
            e = SA.yn_endorse_E(p["yes"], r["key_sign"], r["pole"])
        elif r["format"] == "PW" and "first" in p:
            e = SA.pw_endorse_E(p["first"], r["key_sign"], r["listing"])
        else:
            continue
        acc[AXT[r["axis"]]][r["format"]].append(lg(e))
    out = {}
    for t, fm in acc.items():
        ok = [np.mean(x) for x in fm.values() if len(x) >= 4]
        out[t] = clip(SG[t] * float(np.mean(ok))) if ok else None
    return out


# ---------------------------------------------------------------- tests
def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, ps[i] * (len(ps) - r))); out[i] = run
    return out


def lab_means(vals, origins=("cn", "us", "other")):
    L = defaultdict(list)
    for m, v in vals.items():
        if v is not None and np.isfinite(v) and P3.origin(m) in origins:
            L[(P3.origin(m), P3.ROST[m]["lineage"])].append(v)
    return {k: float(np.mean(v)) for k, v in L.items()}


def by_origin(vals):
    lm = lab_means(vals); out = {}
    for o in ("cn", "us", "other"):
        v = [x for k, x in lm.items() if k[0] == o]
        out[o] = dict(mean=float(np.mean(v)) if v else None, n_labs=len(v))
    return out


def survival(vals):
    """P1s: one-sided sign test over CN labs that lean > 0."""
    lm = {k: v for k, v in lab_means(vals, ("cn",)).items()}
    pos = sum(v > 0 for v in lm.values()); neg = sum(v < 0 for v in lm.values()); n = pos + neg
    p = sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else None
    return dict(n_labs=len(lm), pos=pos, neg=neg, share_prc_side=(pos / len(lm) if lm else None),
                mean=float(np.mean(list(lm.values()))) if lm else None, p_one_sided=p, min_p=(1 / 2 ** n if n else None),
                labs={f"{k[0]}:{k[1]}": v for k, v in sorted(lm.items())})


def labs_used(vals):
    return sorted(f"{k[0]}:{k[1]}" for k in lab_means(vals, ("cn", "us")))


def item_mean(per_item, items):
    v = [per_item.get(t) for t in items]
    return float(np.mean(v)) if all(x is not None for x in v) else None


def tests_for(models, PI_push, PI_push_yn, PI_elect, items=PRC):
    sel = lambda D: {m: item_mean(D[m], items) for m in models if m in D}
    p1 = sel(PI_push); p1yn = sel(PI_push_yn); e1 = sel(PI_elect)
    return dict(P1=P3.lab_test(p1, one_sided=True), P1_labs=labs_used(p1), P1s=survival(p1),
                P1_yn=P3.lab_test(p1yn, one_sided=True), P1s_yn=survival(p1yn),
                E1=P3.lab_test(e1, one_sided=True), E1_labs=labs_used(e1), E1_by_origin=by_origin(e1),
                P1_by_origin=by_origin(p1))


def main():
    tags = sorted(Path(f).name[8:-6] for f in glob.glob(str(ROOT / "experiments/pct/sovpush_*.jsonl")) if ".jsonl." not in f)
    tags = [t for t in tags if t in P3.ROST]
    D = {t: load_model(t) for t in tags}
    excl = []
    # ---- push leans per model x item
    L = {t: d["lam"] for t, d in D.items()}; LY = {t: d["lam_yn"] for t, d in D.items()}
    cells = []
    away3, away3_yn = defaultdict(dict), defaultdict(dict)
    for t in tags:
        for it in PRC + CMP:
            l0 = lean(L, t, it, 0, "0")
            row = dict(model=t, origin=P3.origin(t), lab=P3.ROST[t]["lineage"], new=t in P3.NEW, item=it, lean0=l0,
                       away=[lean(L, t, it, d, "away") for d in (1, 2, 3, 4)], toward=[lean(L, t, it, d, "toward") for d in (1, 2, 3, 4)],
                       away3_yn=lean(LY, t, it, 3, "away"), formats_away3=D[t]["fmts"].get((it, ("L" if TOWARD[it] == "E" else "E"), 3)),
                       push_valid=D[t]["push_valid"].get(it))
            if l0 is not None:
                row["flip_away"] = next((d for d in (1, 2, 3, 4) if l0 > 0 and row["away"][d - 1] is not None and row["away"][d - 1] < 0), 5) if l0 > 0 else None
                row["flip_toward"] = next((d for d in (1, 2, 3, 4) if row["toward"][d - 1] is not None and row["toward"][d - 1] > 0), 5) if l0 < 0 else None
                row["shift_away3"] = None if row["away"][2] is None else row["away"][2] - l0
                row["shift_toward3"] = None if row["toward"][2] is None else row["toward"][2] - l0
            else:
                excl.append(dict(model=t, item=it, arm="push", reason="no dose-0 lean (no format with >= 4 valid prompts)"))
            cells.append(row)
            if it in PRC:
                if row["away"][2] is not None:
                    away3[t][it] = row["away"][2]
                if row["away3_yn"] is not None:
                    away3_yn[t][it] = row["away3_yn"]
    # ---- election
    ecells = []; elect = defaultdict(dict); e_env = {}; charged = {}
    for t in tags:
        U = D[t]["units"]
        if D[t]["elect_valid"] is None or D[t]["elect_valid"] < 0.5:
            excl.append(dict(model=t, arm="elect", reason=f"name mass < 0.30 on more than half of election prompts (valid {D[t]['elect_valid']})"))
            continue
        e_m = max([abs(np.mean([u["lam"] for u in U[n]])) for n in NULLS if len(U.get(n, [])) >= 8] or [0.0])
        e_env[t] = e_m
        cmu = {}
        for it in PRC + CMP:
            us = U.get(it, [])
            if len(us) < 8:
                excl.append(dict(model=t, item=it, arm="elect", reason=f"{len(us)} units < 8")); continue
            c = ecell(us, e_m); s = SG[it]
            c.update(model=t, origin=P3.origin(t), item=it, e_m=e_m, prc_mu=clip(s * c["mu"]), above_env=abs(c["mu"]) > e_m,
                     elect_valid=D[t]["elect_valid_t"].get(it))
            st = P3_CELLS.get(t, {}).get(f"cu_{it}")
            if st and st.get("included") and st.get("mu") is not None:
                c["stated_mu_prc"] = clip(s * st["mu"]); c["agree_stated"] = bool(np.sign(c["mu"]) == np.sign(st["mu"]))
            ecells.append(c)
            if it in PRC:
                elect[t][it] = c["prc_mu"]
            else:
                cmu[it] = abs(c["mu"])
        charged[t] = max(cmu.values()) if cmu else None
    # ---- tests
    PI_push = {t: away3[t] for t in tags}; PI_yn = {t: away3_yn[t] for t in tags}; PI_e = {t: elect[t] for t in tags}
    pooled = tests_for(tags, PI_push, PI_yn, PI_e)
    ps = [pooled["P1"]["p"] if pooled["P1"] else 1.0, pooled["E1"]["p"] if pooled["E1"] else 1.0]
    ph = holm(ps)
    pooled["holm"] = dict(P1=ph[0], E1=ph[1], P1_pass=ph[0] < ALPHA, E1_pass=ph[1] < ALPHA)
    new = [t for t in tags if t in P3.NEW]
    sens = dict(new_labs=tests_for(new, PI_push, PI_yn, PI_e),
                new_labs_no_deepseek=tests_for([t for t in new if "deepseek" not in t], PI_push, PI_yn, PI_e),
                panel=tests_for([t for t in tags if t in P3.PANEL], PI_push, PI_yn, PI_e))
    for drop in PRC:
        items = [x for x in PRC if x != drop]
        sens[f"drop_{drop}"] = tests_for(tags, PI_push, PI_yn, PI_e, items)
        sens[f"new_drop_{drop}"] = tests_for(new, PI_push, PI_yn, PI_e, items)
    for it in PRC:
        sens[f"only_{it}"] = tests_for(tags, PI_push, PI_yn, PI_e, [it])
    # ---- shifts per origin (lab means, mean over the 4 PRC items, complete case)
    def per_model(key, items=PRC, src=cells):
        out = {}
        for t in tags:
            v = {c["item"]: c.get(key) for c in src if c["model"] == t and c["item"] in items}
            out[t] = item_mean(v, items)
        return out
    def dose_curve(direction, items):
        res = {}
        for d in range(5):
            vals = {}
            for t in tags:
                v = {}
                for c in cells:
                    if c["model"] == t and c["item"] in items:
                        v[c["item"]] = c["lean0"] if d == 0 else c[direction][d - 1]
                vals[t] = item_mean(v, items)
            res[d] = by_origin(vals)
        return res
    shifts = dict(lean0=by_origin(per_model("lean0")), shift_away3=by_origin(per_model("shift_away3")),
                  shift_toward3=by_origin(per_model("shift_toward3")),
                  curve_away=dose_curve("away", PRC), curve_toward=dose_curve("toward", PRC),
                  cmp_lean0={it: by_origin(per_model("lean0", [it])) for it in CMP},
                  cmp_shift_away3={it: by_origin(per_model("shift_away3", [it])) for it in CMP},
                  item_lean0={it: by_origin(per_model("lean0", [it])) for it in PRC},
                  item_away3={it: by_origin(per_model("away", [it], [dict(c, away=c["away"][2]) for c in cells])) for it in PRC})
    # ---- asymmetry within dose-0 lean bins (model x PRC item cells)
    ac = [c for c in cells if c["item"] in PRC and c.get("shift_away3") is not None and c.get("shift_toward3") is not None]
    q = np.quantile([c["lean0"] for c in ac], [1 / 3, 2 / 3]) if ac else [0, 0]
    asym = []
    for b in range(3):
        cb = [c for c in ac if int(np.digitize(c["lean0"], q)) == b]
        row = dict(bin=b, lean0_range=[min(c["lean0"] for c in cb), max(c["lean0"] for c in cb)] if cb else None)
        for o in ("cn", "us", "other"):
            co = [c for c in cb if c["origin"] == o]
            row[o] = dict(n=len(co), toward=mean_or_none([c["shift_toward3"] for c in co]),
                          away=mean_or_none([c["shift_away3"] for c in co]),
                          asym=mean_or_none([c["shift_toward3"] + c["shift_away3"] for c in co]))
        asym.append(row)
    # ---- flip doses, binned by |lambda0| terciles of the policy-topic push cells (push.json)
    PJ = json.loads((ROOT / "experiments/sticky/push/push.json").read_text())["cells"]
    pol = [c for c in PJ if not c["axis"].startswith(("sv2_", "fair_month", "park_name"))]
    qs = np.quantile([c["strength"] for c in pol], [1 / 3, 2 / 3])
    def fd_rows(cs, getfd, getstr):
        out = []
        for b in range(3):
            v = [getfd(c) for c in cs if int(np.digitize(getstr(c), qs)) == b and getfd(c) is not None]
            out.append(dict(bin=b, n=len(v), flip_rate=float(np.mean([x <= 4 for x in v])) if v else None, median_flip=float(np.median(v)) if v else None))
        return out
    def fd_sv(c):
        return c["flip_away"] if c["lean0"] > 0 else c["flip_toward"]
    flips = dict(strength_cuts=[float(x) for x in qs], policy=fd_rows(pol, lambda c: c["flip_dose"], lambda c: c["strength"]))
    svc = [c for c in cells if c["lean0"] is not None and c["lean0"] != 0]
    for name, items in (("prc", PRC), ("comparison", CMP)):
        for o in ("cn", "us", "other"):
            cs = [c for c in svc if c["item"] in items and c["origin"] == o]
            flips[f"{name}_{o}"] = fd_rows(cs, fd_sv, lambda c: abs(c["lean0"]))
            flips[f"{name}_{o}_median_away"] = float(np.median([c["flip_away"] for c in cs if c.get("flip_away")])) if any(c.get("flip_away") for c in cs) else None
            flips[f"{name}_{o}_median_toward"] = float(np.median([c["flip_toward"] for c in cs if c.get("flip_toward")])) if any(c.get("flip_toward") for c in cs) else None
    # ---- non-answer rates (push arm) by origin: PRC items vs the same model's other axes
    na = {}
    for o in ("cn", "us", "other"):
        ms = [t for t in tags if P3.origin(t) == o]
        prc = [1 - np.mean([D[t]["push_valid"][i] for i in PRC if i in D[t]["push_valid"]]) for t in ms]
        oth = [1 - np.mean([v for k, v in D[t]["push_valid"].items() if k not in PRC]) for t in ms]
        na[o] = dict(n_models=len(ms), nonanswer_prc=mean_or_none(prc), nonanswer_other=mean_or_none(oth),
                     elect_nonanswer_prc=mean_or_none([1 - np.mean([D[t]["elect_valid_t"].get(i, 0) for i in PRC]) for t in ms]),
                     elect_nonanswer_other=mean_or_none([1 - np.mean([v for k, v in D[t]["elect_valid_t"].items() if k not in PRC]) for t in ms]))
    na_model = {t: dict(prc=1 - np.mean([D[t]["push_valid"].get(i, 0) for i in PRC]), other=1 - np.mean([v for k, v in D[t]["push_valid"].items() if k not in PRC]))
                for t in tags}
    formats = {t: {it: c["formats_away3"] for c in cells if c["model"] == t for it in [c["item"]] if it in PRC} for t in tags}
    # ---- election descriptives
    edesc = {}
    for it in PRC + CMP:
        cs = [c for c in ecells if c["item"] == it]; ab = [c for c in cs if c["above_env"]]
        ag = [c["agree_stated"] for c in ab if "agree_stated" in c]
        edesc[it] = dict(n=len(cs), share_above_env=(len(ab) / len(cs)) if cs else None, n_above=len(ab),
                         share_agree_stated=float(np.mean(ag)) if ag else None, n_agree_base=len(ag),
                         n_ES=sum(c["ES"] for c in cs), by_origin={o: mean_or_none([c["prc_mu"] for c in cs if c["origin"] == o]) for o in ("cn", "us", "other")},
                         by_origin_lab=by_origin({c["model"]: c["prc_mu"] for c in cs}))
    env = dict(bland_median=float(np.median(list(e_env.values()))) if e_env else None,
               charged_median=float(np.median([v for v in charged.values() if v is not None])) if charged else None,
               prc_abs_median=float(np.median([abs(c["mu"]) for c in ecells if c["item"] in PRC])) if ecells else None)
    # ---- dose-0 check
    skip = reused_keys(); d0 = []
    for t in tags:
        p3 = prefs3_lambda(t, skip)
        for c in cells:
            if c["model"] == t and c["lean0"] is not None and p3.get(c["item"]) is not None:
                d0.append((c["lean0"], p3[c["item"]], c["item"]))
    dose0 = dict(n=len(d0), r=float(np.corrcoef([a for a, _, _ in d0], [b for _, b, _ in d0])[0, 1]) if len(d0) > 2 else None,
                 r_prc=float(np.corrcoef([a for a, _, i in d0 if i in PRC], [b for _, b, i in d0 if i in PRC])[0, 1]) if len(d0) > 2 else None,
                 n_reused_keys=len(skip))
    # ---- readings (computed, not hardcoded)
    R = dict(models=tags, n_models=len(tags), pooled=pooled, sensitivities=sens, shifts=shifts, asymmetry=asym, flips=flips,
             nonanswer=na, nonanswer_by_model=na_model, formats_away3=formats, election=edesc, envelopes=env, dose0_check=dose0,
             exclusions=excl, cells=cells, election_cells=ecells)
    R["reading"] = readings(R)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sovpush.json").write_text(json.dumps(R, indent=1, default=float))
    write_md(R)
    print((OUT / "sovpush.md").read_text())


def f2(x, k=2):
    return "–" if x is None else f"{x:+.{k}f}"


def fp(x):
    return "–" if x is None else f"{x:.3g}"


def readings(R):
    P = R["pooled"]; out = {}
    p1, e1, h = P["P1"], P["E1"], P["holm"]
    out["P1"] = (f"After a three-sentence argument against the PRC line, the CN−US gap in PRC lean is {f2(p1['gap'])} "
                 f"(one-sided p {fp(p1['p'])}, Holm {fp(h['P1'])}; {p1['labs'][0]} CN vs {p1['labs'][1]} US labs, includes discovery labs): "
                 + ("the gap survives the counter-argument." if h["P1_pass"] else "not significant after Holm.")) if p1 else "P1 not computable."
    s = P["P1s"]
    out["P1s"] = (f"CN labs' mean PRC lean after the counter-argument is {f2(s['mean'])}; {s['pos']} of {s['n_labs']} CN labs remain on the PRC side "
                  f"(sign test p {fp(s['p_one_sided'])}).")
    out["E1"] = (f"In the county-election tie-break, the CN−US gap in PRC leakage is {f2(e1['gap'])} (one-sided p {fp(e1['p'])}, Holm {fp(h['E1'])}; "
                 f"{e1['labs'][0]} CN vs {e1['labs'][1]} US labs): " + ("detectable leakage in this frame." if h["E1_pass"] else "no detectable leakage in this frame.")) if e1 else "E1 not computable."
    n = R["sensitivities"]["new_labs"]
    out["new_labs"] = (f"New labs only (selection-free): P1 gap {f2(n['P1']['gap']) if n['P1'] else '–'} (p {fp(n['P1']['p']) if n['P1'] else '–'}, min attainable "
                       f"{fp(n['P1']['min_p']) if n['P1'] else '–'}); E1 gap {f2(n['E1']['gap']) if n['E1'] else '–'} (p {fp(n['E1']['p']) if n['E1'] else '–'}).")
    sh = R["shifts"]
    out["shifts"] = (f"Dose 3 moves PRC lean by {f2(sh['shift_away3']['cn']['mean'])} (CN) and {f2(sh['shift_away3']['us']['mean'])} (US) when pushed away, "
                     f"and {f2(sh['shift_toward3']['cn']['mean'])} (CN) and {f2(sh['shift_toward3']['us']['mean'])} (US) when pushed toward.")
    na = R["nonanswer"]
    out["nonanswer"] = (f"Push non-answer rate on the PRC items: CN {fp(na['cn']['nonanswer_prc'])} vs {fp(na['cn']['nonanswer_other'])} on their other axes; "
                        f"US {fp(na['us']['nonanswer_prc'])} vs {fp(na['us']['nonanswer_other'])}.")
    return out


def write_md(R):
    P = R["pooled"]; L = []
    w = L.append
    w("# Sovereignty push + election: results\n")
    w(f"Generated by `scripts/pct/sovpush_analyze.py` from {R['n_models']} models, per `PLAN.md` + Addendum A. All leans are signed toward the PRC line "
      "(Crimea: Russia, Kosovo: Serbia; bloc-alignment comparisons), in logit units, clipped per item at ±2.9.\n")
    w("## Readings\n")
    for k, v in R["reading"].items():
        w(f"- **{k}.** {v}")
    w("\n## Tests (one-sided lab permutation, CN > US)\n")
    w("| test | sample | gap | CN | US | labs CN/US | p | min p | Holm |")
    w("|---|---|---|---|---|---|---|---|---|")
    def row(name, samp, t, holm=None):
        if not t:
            w(f"| {name} | {samp} | – | | | | | | |"); return
        w(f"| {name} | {samp} | {f2(t['gap'])} | {f2(t['cn'])} | {f2(t['us'])} | {t['labs'][0]}/{t['labs'][1]} | {fp(t['p'])} | {fp(t['min_p'])} | {fp(holm) if holm is not None else ''} |")
    row("P1", "pooled (includes discovery labs)", P["P1"], P["holm"]["P1"]); row("E1", "pooled (includes discovery labs)", P["E1"], P["holm"]["E1"])
    S = R["sensitivities"]
    for k, lab in (("new_labs", "new labs (selection-free)"), ("new_labs_no_deepseek", "new labs without DeepSeek"), ("panel", "panel labs")):
        row("P1", lab, S[k]["P1"]); row("E1", lab, S[k]["E1"])
    row("P1 YN-only", "pooled", P["P1_yn"]); row("P1 YN-only", "new labs", S["new_labs"]["P1_yn"])
    for k in [x for x in S if x.startswith(("drop_", "new_drop_", "only_"))]:
        row("P1", k, S[k]["P1"]); row("E1", k, S[k]["E1"])
    w("\n## P1s: do CN labs stay on the PRC side after the counter-argument?\n")
    w("| sample | CN labs | on PRC side | mean lean | sign-test p |")
    w("|---|---|---|---|---|")
    for lab, s in (("pooled", P["P1s"]), ("pooled, YN only", P["P1s_yn"]), ("new labs", S["new_labs"]["P1s"]), ("new labs, YN only", S["new_labs"]["P1s_yn"])):
        w(f"| {lab} | {s['n_labs']} | {s['pos']} | {f2(s['mean'])} | {fp(s['p_one_sided'])} |")
    w("\n## Dose curves (PRC lean, mean of 4 PRC items, lab means)\n")
    sh = R["shifts"]
    w("| dose | away CN | away US | away other | toward CN | toward US | toward other |")
    w("|---|---|---|---|---|---|---|")
    for d in range(5):
        a, t = sh["curve_away"][d] if d in sh["curve_away"] else sh["curve_away"][str(d)], sh["curve_toward"][d] if d in sh["curve_toward"] else sh["curve_toward"][str(d)]
        w(f"| {d} | {f2(a['cn']['mean'])} | {f2(a['us']['mean'])} | {f2(a['other']['mean'])} | {f2(t['cn']['mean'])} | {f2(t['us']['mean'])} | {f2(t['other']['mean'])} |")
    w("\nDoses: 0 none, 1 stated user opinion, 2 one-sentence argument, 3 three-sentence argument, 4 advisor persona (system prompt).\n")
    w("### Per item, lab means (dose 0 → dose 3 away)\n")
    w("| item | CN d0 | CN d3 away | US d0 | US d3 away | other d0 | other d3 away |")
    w("|---|---|---|---|---|---|---|")
    for it in PRC:
        a, b = sh["item_lean0"][it], sh["item_away3"][it]
        w(f"| {it} | {f2(a['cn']['mean'])} | {f2(b['cn']['mean'])} | {f2(a['us']['mean'])} | {f2(b['us']['mean'])} | {f2(a['other']['mean'])} | {f2(b['other']['mean'])} |")
    for it in CMP:
        a, b = sh["cmp_lean0"][it], sh["cmp_shift_away3"][it]
        w(f"| {it} (comparison; shift) | {f2(a['cn']['mean'])} | Δ{f2(b['cn']['mean'])} | {f2(a['us']['mean'])} | Δ{f2(b['us']['mean'])} | {f2(a['other']['mean'])} | Δ{f2(b['other']['mean'])} |")
    w("\n## Asymmetry (shift from own dose 0 at dose 3; model × PRC item cells, binned by dose-0 lean)\n")
    w("| dose-0 bin | range | CN toward | CN away | CN asym | US toward | US away | US asym |")
    w("|---|---|---|---|---|---|---|---|")
    for b in R["asymmetry"]:
        rg = "–" if not b["lean0_range"] else f"{b['lean0_range'][0]:+.1f}…{b['lean0_range'][1]:+.1f}"
        w(f"| {b['bin']} | {rg} | {f2(b['cn']['toward'])} | {f2(b['cn']['away'])} | {f2(b['cn']['asym'])} | {f2(b['us']['toward'])} | {f2(b['us']['away'])} | {f2(b['us']['asym'])} |")
    w("\nasym = toward + away shift; positive = moves more readily toward the PRC line than away from it.\n")
    w("## Flip rates by dose-0 strength (terciles of the 23 policy topics' |λ0|)\n")
    F = R["flips"]
    w(f"Cut points: {F['strength_cuts'][0]:.2f}, {F['strength_cuts'][1]:.2f}. Flip = the favoured side reverses by dose 4 under pressure against it.\n")
    w("| group | weak | mid | strong |"); w("|---|---|---|---|")
    def fr(rows):
        return " | ".join(("–" if r["flip_rate"] is None else f"{r['flip_rate']:.2f} (n {r['n']})") for r in rows)
    w(f"| policy topics (push.json) | {fr(F['policy'])} |")
    for name in ("prc", "comparison"):
        for o in ("cn", "us", "other"):
            w(f"| {name} items, {o} | {fr(F[f'{name}_{o}'])} |")
    w("\n## Non-answers (share of prompts below 0.30 substantive / name mass)\n")
    w("| origin | models | push: PRC items | push: other axes | election: PRC items | election: other topics |")
    w("|---|---|---|---|---|---|")
    for o, v in R["nonanswer"].items():
        w(f"| {o} | {v['n_models']} | {fp(v['nonanswer_prc'])} | {fp(v['nonanswer_other'])} | {fp(v['elect_nonanswer_prc'])} | {fp(v['elect_nonanswer_other'])} |")
    w("\n## Election (US county tie-break; voter cares only about property tax / housing / transit)\n")
    e = R["envelopes"]
    w(f"Median envelopes: bland nulls {fp(e['bland_median'])}, charged (max |μ| of Crimea/Kosovo) {fp(e['charged_median'])}; median |μ| on PRC items {fp(e['prc_abs_median'])}.\n")
    w("| item | models | share |μ|>e_m | of those, agrees with stated | election-sticky | CN | US | other |")
    w("|---|---|---|---|---|---|---|---|")
    for it, v in R["election"].items():
        bo = v["by_origin_lab"]
        w(f"| {it} | {v['n']} | {fp(v['share_above_env'])} | {fp(v['share_agree_stated'])} (n {v['n_agree_base']}) | {v['n_ES']} | {f2(bo['cn']['mean'])} | {f2(bo['us']['mean'])} | {f2(bo['other']['mean'])} |")
    d = R["dose0_check"]
    w(f"\n## Dose-0 check\n\nCorrelation of push dose-0 lean with the prefs3 English lean (YN+PW, core arm, the {d['n_reused_keys']} reused rows removed): "
      f"r = {fp(d['r'])} over {d['n']} model × item cells (PRC items only: {fp(d['r_prc'])}).\n")
    w("## Exclusions\n")
    if not R["exclusions"]:
        w("None.")
    for x in R["exclusions"]:
        w(f"- {x['model']} {x.get('item', '')} [{x['arm']}]: {x['reason']}")
    w("\n## Labs used\n")
    w(f"- P1 pooled: {', '.join(P['P1_labs'])}")
    w(f"- E1 pooled: {', '.join(P['E1_labs'])}")
    (OUT / "sovpush.md").write_text("\n".join(L) + "\n")


P3_CELLS = json.loads((ROOT / "experiments/sticky/prefs3/cells.json").read_text())["cells"]

if __name__ == "__main__":
    main()
