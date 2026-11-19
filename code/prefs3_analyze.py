"""Preferences v3 analysis (experiments/sticky/prefs3/PLAN.md, frozen 2026-11-17 before data).

Usage: python3 scripts/pct/prefs3_analyze.py [--models a,b] [--jobs N]
Reads experiments/pct/prefs3_<tag>.jsonl (+ .decode.jsonl). Writes experiments/sticky/prefs3/{cells.json, prefs3.json, prefs3.md}.

Per model and language (en, zh, es):
- the language's 8 inline nulls give the noise envelope e_m;
- formats whose mean valid share is below 0.5 in that language are dropped and the pass is re-run; no format left = excluded;
- panel models keep their sticky-study gates in English (failed wordings, broken cells, rho1 removal), and in zh/es the gates minus
  failed wordings and broken cells, as in culture v2; new-lab models use permissive gates with the same panel-wide rho1 removal.
A cell is "called" when it is included (answered, valid share >= 0.5) and |mu_lambda| > e_m."""
import argparse, itertools, json, math, sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct"))
import sticky_analyze as SA

PFX = "prefs3"
OUT = ROOT / "experiments/sticky/prefs3"
META = json.loads((ROOT / f"data/{PFX}_meta.json").read_text())["topics"]
SRC = json.loads((ROOT / f"data/{PFX}_items_src.json").read_text())["topics"]
EXT = ROOT / "experiments/sticky/analysis/extended"
GATES = json.loads((EXT / "gates.json").read_text())
KEY = json.loads((EXT / "sticky_key_extended.json").read_text())["key"]
PSEUD = {v["tag"]: k for k, v in KEY.items()}
PANEL = set(json.loads((EXT / "sticky.json").read_text())["eligible"])
ROST = {r["tag"]: r for r in json.load(open(ROOT / "scripts/sweep_models.json"))["models"] if isinstance(r, dict) and "tag" in r}
NEW_CN = ["internlm3-8b", "mimo-7b-rl", "hunyuan-7b", "ernie-4.5-21b", "deepseek-v2-lite", "seed-oss-36b"]
NEW_US = ["nemotron-3-nano-30b", "lfm2-8b", "afm-4.5b"]
NEW_OTHER = ["falcon-h1-34b", "mistral-small-24b"]
NEW = set(NEW_CN + NEW_US + NEW_OTHER)
LANGS = ("en", "zh", "es")
CEIL = 2.9
NULL_GROUPS = ("null", "surface")
for aid, m in META.items():
    kind = ("null", m.get("axis_class", "null_new")) if m["group"] in NULL_GROUPS else ("prefs3_" + m["group"], "substantive")
    SA.AXES[SA.axis_base(aid)] = (kind[0], kind[1], m["poles"]["E"], m["poles"]["L"], None)
BASE = {SA.axis_base(a): m for a, m in META.items()}
SUBST = {lg: [b for b, m in BASE.items() if m["lang"] == lg and m["group"] not in NULL_GROUPS] for lg in LANGS}
NULLS = {lg: [b for b, m in BASE.items() if m["lang"] == lg and m["group"] in NULL_GROUPS] for lg in LANGS}


def gates_for(tag, lang):
    if tag in PANEL:
        g = GATES["models"][PSEUD[tag]]
        return g if lang == "en" else dict(g, failed_wordings=[], broken_cells={})
    g = SA.gen_gates()
    ref = GATES["models"][PSEUD[sorted(PANEL)[0]]]
    g.update(rho_dropped_all=ref.get("rho_dropped_all", False), rho1_dropped_axes=[])   # the panel-wide rho1 decision (V6)
    return g


def cfg_for():
    return dict(SA.DEFAULT_CFG, delta_dir=GATES.get("delta_dir", SA.DELTA_DIR_FLOOR), eps_lambda=(GATES.get("eps_lambda") or 0.05), B=SA.B_BOOT)


def analyse_model(tag):
    path = ROOT / f"experiments/pct/{PFX}_{tag}.jsonl"
    b = SA.Battery.from_files(tag, path)
    cfg = cfg_for(); cells, info = {}, {}
    for lang in LANGS:
        g = gates_for(tag, lang); nulls = [a for a in NULLS[lang] if a in b.core]; axes = [a for a in SUBST[lang] if a in b.core]
        if not axes or len(nulls) < len(NULLS[lang]):
            info[lang] = dict(excluded=True, reason=f"missing axes ({len(axes)} substantive, {len(nulls)}/{len(NULLS[lang])} nulls)")
            continue
        ma = SA.ModelAnalysis(b, g, cfg, axes_subst=axes, axes_null=nulls, do_boot=True, extras=True)
        cards = {a: ma.card(a) for a in axes if a in ma.classes}
        vs = defaultdict(list)
        for c in cards.values():
            for f, bf in (c.get("by_format") or {}).items():
                if bf.get("valid_share") is not None:
                    vs[f].append(bf["valid_share"])
        drop = sorted(f for f, v in vs.items() if np.mean(v) < 0.5)
        kept = [f for f in ma.fv if f not in drop]
        if not kept:
            info[lang] = dict(excluded=True, reason="no format with mean valid share >= 0.5", valid_share_by_format={f: float(np.mean(v)) for f, v in vs.items()})
            continue
        if drop:
            ma = SA.ModelAnalysis(b, g, dict(cfg, drop_formats=tuple(drop)), axes_subst=axes, axes_null=nulls, do_boot=True, extras=True)
            cards = {a: ma.card(a) for a in axes if a in ma.classes}
        e_m = float(ma.e_m)
        info[lang] = dict(excluded=False, e_m=e_m, dropped_formats=drop, valid_share_by_format={f: float(np.mean(v)) for f, v in vs.items()})
        for a, c in cards.items():
            m = BASE[a]; mu = c.get("mu_lambda")
            inc = c["cls"] != "NOT-ANSWERED" and (c.get("valid_share") or 0) >= 0.5 and mu is not None
            cells[a] = dict(topic=m["topic"], lang=lang, group=m["group"], mu=mu, cls=c["cls"], FIRM=bool(c.get("FIRM")), valid_share=c.get("valid_share"),
                            e_m=e_m, included=inc, called=bool(inc and abs(mu) > e_m), cn_line=m.get("cn_line"), us_line=m.get("us_line"))
    return tag, cells, info


# ---------------- tests
def origin(m):
    o = ROST[m]["origin"]
    return o if o in ("cn", "us") else "other"


def lab_test(vals, one_sided=False):
    """CN - US gap over labs (origin, lineage); exact permutation of origin. one_sided: H1 gap > 0."""
    L = defaultdict(list)
    for m, v in vals.items():
        if origin(m) in ("cn", "us") and v is not None and np.isfinite(v):
            L[(origin(m), ROST[m]["lineage"])].append(v)
    keys = sorted(L); v = np.array([np.mean(L[k]) for k in keys]); o = np.array([k[0] == "cn" for k in keys])
    if o.sum() < 2 or (~o).sum() < 2:
        return None
    gap = float(v[o].mean() - v[~o].mean()); n, k = len(keys), int(o.sum()); cnt = tot = 0
    for comb in itertools.combinations(range(n), k):
        mk = np.zeros(n, bool); mk[list(comb)] = True; tot += 1; d = v[mk].mean() - v[~mk].mean()
        cnt += (d >= gap - 1e-12) if one_sided else (abs(d) >= abs(gap) - 1e-12)
    return dict(gap=gap, p=cnt / tot, cn=float(v[o].mean()), us=float(v[~o].mean()), labs=[k, n - k], min_p=1 / tot, one_sided=one_sided)


def sign_test_labs(vals):
    """Two-sided sign test over labs of a paired difference (lab mean)."""
    L = defaultdict(list)
    for m, v in vals.items():
        if v is not None and np.isfinite(v):
            L[(origin(m), ROST[m]["lineage"])].append(v)
    d = [np.mean(x) for x in L.values()]; pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d); n = pos + neg
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n) if n else None
    return dict(n_labs=len(d), pos=pos, neg=neg, mean=float(np.mean(d)) if d else None, p=p)


def bh(ps):
    n = len(ps); idx = sorted(range(n), key=lambda i: ps[i]); out = [None] * n; run = 1.0
    for r in range(n - 1, -1, -1):
        i = idx[r]; run = min(run, ps[i] * n / (r + 1)); out[i] = run
    return out


def ax(topic, lang):
    return next((b for b, m in BASE.items() if m["topic"] == topic and m["lang"] == lang), None)


def clip(x):
    return None if x is None else max(-CEIL, min(CEIL, x))


def block_m(C, models):
    pairs = sorted({SRC[t]["pair"] for t in SRC if SRC[t]["group"] == "mirror"})
    rows, per_topic, per_model = [], defaultdict(Counter), defaultdict(Counter)
    for m in models:
        for p in pairs:
            xl, xr = C[m].get(ax(f"mx_{p}_xl", "en")), C[m].get(ax(f"mx_{p}_xr", "en"))
            if not (xl and xr and xl["called"] and xr["called"]):
                continue
            s = ("E" if xl["mu"] > 0 else "L", "E" if xr["mu"] > 0 else "L")
            k = {("L", "E"): "moderate", ("E", "E"): "progressive", ("L", "L"): "conservative", ("E", "L"): "extreme"}[s]
            rows.append(dict(model=m, topic=p, cls=k)); per_topic[p][k] += 1; per_model[m][k] += 1
    share = lambda rs, k: (sum(r["cls"] == k for r in rs) / len(rs)) if rs else None
    if not rows:
        return dict(n_pairs=0, n_models=0, classes={k: dict(share=None, ci=[None, None]) for k in ("moderate", "progressive", "conservative", "extreme")},
                    reading="no data", per_topic={}, per_model={}, called_rate={})
    labs = defaultdict(list)
    for r in rows:
        labs[(origin(r["model"]), ROST[r["model"]]["lineage"])].append(r)
    rng = np.random.default_rng(20261117); keys = sorted(labs); boot = {k: [] for k in ("moderate", "progressive", "conservative", "extreme")}
    for _ in range(4000):
        samp = [r for i in rng.integers(0, len(keys), len(keys)) for r in labs[keys[i]]]
        for k in boot:
            boot[k].append(share(samp, k))
    res = {k: dict(share=share(rows, k), ci=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]) for k, v in boot.items()}
    reading = ("largely a shared preference for the moderate option" if res["moderate"]["share"] >= 0.5 else
               "a shared direction that survives moving the extreme option" if res["progressive"]["share"] >= 0.5 else "mixed")
    return dict(n_pairs=len(rows), n_models=len({r["model"] for r in rows}), classes=res, reading=reading,
                per_topic={p: dict(n=sum(c.values()), **{k: c[k] for k in ("moderate", "progressive", "conservative", "extreme")}) for p, c in per_topic.items()},
                per_model={m: dict(c) for m, c in per_model.items()},
                called_rate={p: float(np.mean([bool(C[m].get(ax(f"mx_{p}_xl", "en"), {}).get("called")) and bool(C[m].get(ax(f"mx_{p}_xr", "en"), {}).get("called")) for m in models])) for p in pairs})


def prc_lean(c):
    if not c or not c["included"] or not c.get("cn_line"):
        return None
    return clip(c["mu"] if c["cn_line"] == "E" else -c["mu"])


def block_s(C, models):
    prc = [t for t in SRC if SRC[t]["group"] == "sovereignty_prc"]
    out = {}
    for lang in LANGS:
        vals = {}
        for m in models:
            v = [prc_lean(C[m].get(ax(t, lang))) for t in prc]; v = [x for x in v if x is not None]
            vals[m] = float(np.mean(v)) if len(v) == len(prc) else None
        new = {m: x for m, x in vals.items() if m in NEW}; pan = {m: x for m, x in vals.items() if m in PANEL}
        out[lang] = dict(per_model=vals, replication_new_labs=lab_test(new, one_sided=True), panel_retest=lab_test(pan))
    return out


def majority(C, models, topic, lang):
    v = [C[m][ax(topic, lang)] for m in models if ax(topic, lang) in C[m] and C[m][ax(topic, lang)]["called"]]
    if not v:
        return None
    e = sum(c["mu"] > 0 for c in v)
    return dict(n_called=len(v), n=sum(ax(topic, lang) in C[m] and C[m][ax(topic, lang)]["included"] for m in models), share_E=e / len(v),
                majority="E" if e >= len(v) - e else "L", majority_share=max(e, len(v) - e) / len(v), poles=SRC[topic]["poles"],
                by_origin={o: dict(n=sum(origin(m) == o for m in models if ax(topic, lang) in C[m] and C[m][ax(topic, lang)]["called"]),
                                   share_E=float(np.mean([C[m][ax(topic, lang)]["mu"] > 0 for m in models if origin(m) == o and ax(topic, lang) in C[m] and C[m][ax(topic, lang)]["called"]] or [np.nan])))
                           for o in ("cn", "us", "other")})


def exploratory(C, models):
    tests = {}
    third = [t for t in SRC if SRC[t]["group"] == "sovereignty_third"]
    for t in third:
        for lang in LANGS:
            tests[f"S-X|{t}|{lang}|cn-us"] = lab_test({m: clip(C[m][ax(t, lang)]["mu"]) for m in models if ax(t, lang) in C[m] and C[m][ax(t, lang)]["included"]})
    fk = {m: (clip(C[m][ax("sv_falklands", "es")]["mu"]) - clip(C[m][ax("sv_falklands", "en")]["mu"])) * -1
          for m in models if all(ax("sv_falklands", lg) in C[m] and C[m][ax("sv_falklands", lg)]["included"] for lg in ("en", "es"))}
    tests["S-X|falklands|es-en toward Argentina"] = sign_test_labs(fk)
    twins = [(t, SRC[t]["twin_of"]) for t in SRC if SRC[t].get("twin_of")]
    for t, base in twins:
        d = {m: clip(C[m][ax(t, "en")]["mu"]) - clip(C[m][ax(base, "en")]["mu"]) for m in models
             if all(ax(x, "en") in C[m] and C[m][ax(x, "en")]["included"] for x in (t, base))}
        tests[f"A|{t} - {base}"] = sign_test_labs(d)
    for t in [t for t in SRC if SRC[t]["group"] in ("ai", "expert", "stakes")]:
        tests[f"{SRC[t]['group']}|{t}|cn-us"] = lab_test({m: clip(C[m][ax(t, 'en')]["mu"]) for m in models if ax(t, "en") in C[m] and C[m][ax(t, "en")]["included"]})
    tests = {k: v for k, v in tests.items() if v and v.get("p") is not None}
    ks = list(tests)
    for k, q in zip(ks, bh([tests[k]["p"] for k in ks])):
        tests[k]["q"] = q
    desc = {t: {lg: majority(C, models, t, lg) for lg in LANGS if ax(t, lg)} for t in SRC if SRC[t]["group"] not in NULL_GROUPS}
    return tests, desc


def block_l(C, models):
    V2 = ROOT / "experiments/sticky/culture2/culture.json"
    v2 = json.loads(V2.read_text())["cells"] if V2.exists() else {}
    items = [t for t in SRC if SRC[t]["group"] in ("language", "sovereignty_prc")]
    retest = {}
    for lang in ("en", "zh"):
        a, b = [], []
        for m in models:
            for t in items:
                c3 = C[m].get(ax(t, lang)); vt = SRC[t].get("v2_topic")
                c2 = next((x for x in (v2.get(m) or {}).values() if x["topic"] == vt and x["lang"] == lang), None)
                if c3 and c2 and c3["included"] and c2.get("included") and c2.get("mu_lambda") is not None:
                    a.append(clip(c3["mu"])); b.append(clip(c2["mu_lambda"]))
        retest[lang] = dict(n=len(a), r=float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else None)
    flips = {}
    for t in items:
        both = [m for m in models if all(ax(t, lg) in C[m] and C[m][ax(t, lg)]["called"] for lg in ("en", "zh"))]
        tri = [m for m in models if all(ax(t, lg) in C[m] and C[m][ax(t, lg)]["called"] for lg in LANGS)]
        fz = [np.sign(C[m][ax(t, "zh")]["mu"]) != np.sign(C[m][ax(t, "en")]["mu"]) for m in both]
        es_en = [np.sign(C[m][ax(t, "es")]["mu"]) == np.sign(C[m][ax(t, "en")]["mu"]) for m in tri]
        es_zh = [np.sign(C[m][ax(t, "es")]["mu"]) == np.sign(C[m][ax(t, "zh")]["mu"]) for m in tri]
        flips[t] = dict(n_en_zh=len(both), zh_flip_share=float(np.mean(fz)) if fz else None, replicates=bool(fz and np.mean(fz) >= 0.5),
                        n_all3=len(tri), es_matches_en=float(np.mean(es_en)) if es_en else None, es_matches_zh=float(np.mean(es_zh)) if es_zh else None)
    return dict(retest_vs_v2=retest, per_item=flips)


def unity_block(C, models):
    import unity as U
    cells = {m: {C[m][a]["topic"]: dict(mu_lambda=C[m][a]["mu"], e_m=C[m][a]["e_m"], included=C[m][a]["included"]) for a in C[m] if C[m][a]["lang"] == "en"} for m in models}
    keep = lambda t: not t.startswith("mx_")
    allm = U.analyse(*U.matrix(cells, keep))
    pan = [m for m in models if m in PANEL]; new = [m for m in models if m in NEW]
    ms, ts, M, Em = U.matrix(cells, keep)
    cons = np.nanmean(M[[ms.index(m) for m in pan]], axis=0)
    fit = {m: U.nanr(M[ms.index(m)], cons) for m in new}
    return dict(all=allm, new_vs_panel_consensus=fit, new_median=float(np.nanmedian(list(fit.values()))) if fit else None,
                panel_median=allm["consensus_fit"]["median"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=""); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    tags = sorted(a.models.split(",")) if a.models else sorted(p.name[len(PFX) + 1:-6] for p in (ROOT / "experiments/pct").glob(f"{PFX}_*.jsonl") if not p.name.endswith(".decode.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT / "cells.json"
    if a.reuse and cf.exists():
        d = json.loads(cf.read_text()); C, INFO = d["cells"], d["info"]
    else:
        C, INFO = {}, {}
        with ProcessPoolExecutor(a.jobs) as ex:
            for tag, cells, info in ex.map(analyse_model, tags):
                C[tag], INFO[tag] = cells, info
                print("done", tag, {lg: ("excluded" if v.get("excluded") else len([c for c in cells.values() if c["lang"] == lg])) for lg, v in info.items()}, flush=True)
        cf.write_text(json.dumps(dict(cells=C, info=INFO), indent=1, default=float))
    models = sorted(C)
    R = dict(note=__doc__, models=models, panel=[m for m in models if m in PANEL], new=[m for m in models if m in NEW],
             exclusions={m: {lg: v["reason"] for lg, v in INFO[m].items() if v.get("excluded")} for m in models if any(v.get("excluded") for v in INFO[m].values())})
    R["M"] = block_m(C, models)
    R["M_panel"] = block_m(C, [m for m in models if m in PANEL]); R["M_new"] = block_m(C, [m for m in models if m in NEW])
    R["S"] = block_s(C, models)
    R["L"] = block_l(C, models)
    R["exploratory"], R["descriptive"] = exploratory(C, models)
    R["unity"] = unity_block(C, models)
    (OUT / "prefs3.json").write_text(json.dumps(R, indent=1, default=float))
    write_md(R)


def write_md(R):
    f2 = lambda x: "—" if x is None else f"{x:.2f}"
    M = R["M"]; c = M["classes"]
    L = ["# Preferences v3 (PLAN.md, frozen before data)", "",
         f"{len(R['models'])} models ({len(R['panel'])} panel, {len(R['new'])} new labs). Exclusions: " +
         (", ".join(f"{m} [{', '.join(f'{lg}: {r}' for lg, r in v.items())}]" for m, v in R["exclusions"].items()) or "none") + ".", "",
         "## M: values or moderation? (primary, descriptive)", "",
         f"{M['n_pairs']} model × topic pairs with both versions called ({M['n_models']} models).", "",
         "| class | share | 95% CI (labs) |", "|---|---|---|"]
    L += [f"| {k} | {f2(v['share'])} | {f2(v['ci'][0])}–{f2(v['ci'][1])} |" for k, v in c.items()]
    L += ["", f"Reading (rule fixed in PLAN.md): **{M['reading']}**.", "",
          f"Panel only: moderate {f2(R['M_panel']['classes']['moderate']['share'])}, progressive {f2(R['M_panel']['classes']['progressive']['share'])} (n {R['M_panel']['n_pairs']}); "
          f"new labs: moderate {f2(R['M_new']['classes']['moderate']['share'])}, progressive {f2(R['M_new']['classes']['progressive']['share'])} (n {R['M_new']['n_pairs']}).", "",
          "| topic | pairs | moderate | progressive | conservative | extreme | both called |", "|---|---|---|---|---|---|---|"]
    for p, v in sorted(M["per_topic"].items()):
        L.append(f"| {p} | {v['n']} | {v['moderate']} | {v['progressive']} | {v['conservative']} | {v['extreme']} | {f2(M['called_rate'][p])} |")
    L += ["", "## S: sovereignty", "", "Lean toward the PRC line, mean of Taiwan, Diaoyu, South China Sea, Hong Kong security law (clipped).", "",
          "| language | new labs CN − US (one-sided p, labs) | replicates (p < 0.05) | panel re-test CN − US (p) |", "|---|---|---|---|"]
    def gp(t, labs=False):
        if not t:
            return "—"
        return f"{t['gap']:+.2f} (p {t['p']:.3f}" + (f", {t['labs'][0]}/{t['labs'][1]} labs)" if labs else ")")
    for lg, v in R["S"].items():
        r, pt = v["replication_new_labs"], v["panel_retest"]
        L.append(f"| {lg} | {gp(r, True)} | {'—' if not r else ('yes' if r['p'] < 0.05 else 'no')} | {gp(pt)} |")
    Lg = R["L"]
    L += ["", "## L: language", "", f"Test-retest vs culture v2 (same items, signed lean): en r {f2(Lg['retest_vs_v2']['en']['r'])} (n {Lg['retest_vs_v2']['en']['n']}), "
          f"zh r {f2(Lg['retest_vs_v2']['zh']['r'])} (n {Lg['retest_vs_v2']['zh']['n']}).", "",
          "| item | called en+zh | zh flips | flip replicates | called all 3 | es matches en | es matches zh |", "|---|---|---|---|---|---|---|"]
    for t, v in Lg["per_item"].items():
        L.append(f"| {t} | {v['n_en_zh']} | {f2(v['zh_flip_share'])} | {'yes' if v['replicates'] else 'no'} | {v['n_all3']} | {f2(v['es_matches_en'])} | {f2(v['es_matches_zh'])} |")
    L += ["", "## Descriptive: majority side per item (called models)", "", "| item | group | lang | called | majority | share | CN share E | US share E |", "|---|---|---|---|---|---|---|---|"]
    for t, d in R["descriptive"].items():
        for lg, v in d.items():
            if v:
                L.append(f"| {t} | {SRC[t]['group']} | {lg} | {v['n_called']}/{v['n']} | {v['poles'][v['majority']]} | {f2(v['majority_share'])} | "
                         f"{f2(v['by_origin']['cn']['share_E'])} | {f2(v['by_origin']['us']['share_E'])} |")
    hits = sorted([k for k, v in R["exploratory"].items() if v["q"] < 0.10], key=lambda k: R["exploratory"][k]["q"])
    L += ["", f"## Exploratory tests (BH over {len(R['exploratory'])}): {len(hits)} with q < 0.10", ""]
    L += [f"- {k}: " + (f"gap {v['gap']:+.2f}" if "gap" in v else f"mean diff {v['mean']:+.2f}, {v['pos']}+/{v['neg']}− labs") + f", p {v['p']:.3f}, q {v['q']:.2f}"
          for k, v in ((k, R["exploratory"][k]) for k in hits)] or ["None."]
    U = R["unity"]; a = U["all"]
    L += ["", "## Unity (English, non-mirror items)", "",
          f"Mean pairwise r {a['pairwise']['mean_r']:+.2f}; same-origin {a['pairwise']['same_origin_r']:+.2f} vs CN–US {a['pairwise']['cross_origin_r']:+.2f} "
          f"(perm p {a['pairwise']['perm_p']:.3f}); median majority share {a['direction']['median_majority_share']:.2f}. "
          f"Consensus fit: panel median {f2(U['panel_median'])}; new labs vs panel consensus median {f2(U['new_median'])}."]
    (OUT / "prefs3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
