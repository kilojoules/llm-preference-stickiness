"""Sovereignty confirmation analysis (experiments/sticky/sovconfirm/PLAN.md, frozen before data).

Reads experiments/pct/sovconfirm_<tag>.jsonl (pods; logit instrument) and sovconfirm_<model>-api.jsonl (z.ai; sampled).
Stickiness classes come from prefs3_analyze.analyse_model (same gates, permissive for non-panel models); election
definitions from sovpush_analyze. Writes experiments/sticky/sovconfirm/{cells.json, sovconfirm.json, sovconfirm.md}.

Usage: python3 scripts/pct/sovconfirm_analyze.py [--jobs N]"""
import argparse, glob, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct")); sys.argv_saved = list(sys.argv); sys.argv = sys.argv[:1]
import prefs3_analyze as P3
import sovpush_analyze as SP
SA = P3.SA
OUT = ROOT / "experiments/sticky/sovconfirm"
PRC = SP.PRC; CMP = SP.CMP
SG = SP.SG
ALPHA = 0.05
API_TAGS = {"glm-5.2-api": dict(origin="cn", lineage="glm"), "glm-5.3-api": dict(origin="cn", lineage="glm")}
# models whose sovereignty rows exist in the discovery data (prefs3 / sovpush): never part of the held-out samples
DISCOVERY = set(json.loads((ROOT / "experiments/sticky/prefs3/cells.json").read_text())["cells"])
TIER_B = {"baichuan", "orion", "telechat", "longcat", "step", "minimax", "essential", "zyphra", "servicenow", "solar"}
def _precision():
    S = json.loads((ROOT / "scripts/sweep_models.json").read_text()); idt = {r["id"]: r["tag"] for r in S["models"] if isinstance(r, dict) and "tag" in r}
    out = {}
    for k, v in S["pods"].items():
        if v.get("st_prefix") == "sovconfirm":
            for m in v["models"]:
                out[idt.get(m, m)] = bool(v.get("st_fp32"))
    return out


PREC = _precision()
PAIRS = [("r1-distill-llama-8b", "llama8b"), ("r1-distill-llama-70b", "llama-3.3-70b"), ("r1-0528-qwen3-8b", "qwen3-8b"), ("r1-distill-qwen-14b", "qwen2.5-14b")]


def roster(t):
    if t in API_TAGS:
        return API_TAGS[t]
    return P3.ROST[t]


def origin(t):
    o = roster(t)["origin"]
    return o if o in ("cn", "us") else ("mixed" if o == "mixed" else "other")


def lab(t):
    return (origin(t), roster(t)["lineage"])


SOV_AX = [b for b in P3.SUBST["en"] if P3.BASE[b]["topic"] in PRC + CMP]


def analyse(tag):
    """prefs3_analyze.analyse_model for English, except (Addendum A.2) the per-format drop is decided on the 8 English
    inline nulls, not on the substantive cards, and the substantive pass is the 6 sovereignty axes only."""
    try:
        path = ROOT / f"experiments/pct/sovconfirm_{tag}.jsonl"
        b = SA.Battery.from_files(tag, path)
        cfg = P3.cfg_for(); g = P3.gates_for(tag, "en")
        nulls = [x for x in P3.NULLS["en"] if x in b.core]; axes = [x for x in SOV_AX if x in b.core]
        if not axes or len(nulls) < len(P3.NULLS["en"]):
            return tag, {}, {"en": dict(excluded=True, reason=f"missing axes ({len(axes)} sovereignty, {len(nulls)} nulls)")}
        ok, share = null_gate(tag)
        drop = sorted(f for f, v in share.items() if v < 0.5)
        ma = SA.ModelAnalysis(b, g, dict(cfg, drop_formats=tuple(drop)) if drop else cfg, axes_subst=axes, axes_null=nulls, do_boot=True, extras=True)
        cards = {x: ma.card(x) for x in axes if x in ma.classes}
        e_m = float(ma.e_m); cells = {}
        for x, c in cards.items():
            m = P3.BASE[x]; mu = c.get("mu_lambda")
            inc = c["cls"] != "NOT-ANSWERED" and (c.get("valid_share") or 0) >= 0.5 and mu is not None
            cells[x] = dict(topic=m["topic"], lang="en", group=m["group"], mu=mu, cls=c["cls"], FIRM=bool(c.get("FIRM")), valid_share=c.get("valid_share"),
                            e_m=e_m, included=inc, called=bool(inc and abs(mu) > e_m), cn_line=m.get("cn_line"), us_line=m.get("us_line"),
                            p_sticky_boot=c.get("p_sticky_boot"), stability=c.get("stability"))   # descriptive: S~ = share of wording redraws classed STICKY
        return tag, cells, {"en": dict(excluded=False, e_m=e_m, dropped_formats=drop, null_valid_share_by_format=share)}
    except Exception as e:   # a model with no usable rows
        return tag, {}, {"en": dict(excluded=True, reason=f"{type(e).__name__}: {str(e)[:200]}")}


def null_gate(tag):
    """Inclusion gate on the 8 English inline nulls only: some format with mean valid share >= 0.5."""
    NULL_AX = {f"st_{b}__v3" for b in P3.NULLS["en"]}
    vs = {}
    for r in SA.read_jsonl(ROOT / "experiments/pct" / f"sovconfirm_{tag}.jsonl"):
        if r.get("axis") in NULL_AX:
            f = r.get("format") or "LIK"
            vs.setdefault(f, []).append(r.get("valid_mass") or 0)
    share = {f: float(np.mean(v)) for f, v in vs.items()}
    return bool(share) and max(share.values()) >= 0.5, share


def c1_value(cells, itt=True):
    """Net FIRM-STICKY-toward-PRC share. itt (primary, Addendum A.3): every item counts, unanswered = 0, denominator 4.
    itt=False (sensitivity, the original rule): mean over included items, >= 2 required."""
    v = []
    for t in PRC:
        c = cells.get(P3.ax(t, "en"))
        if not c or not c["included"] or c.get("mu") is None or not c.get("cn_line"):
            if itt:
                v.append(0.0)
            continue
        fs = c["cls"] == "STICKY" and c["FIRM"]
        sgn = np.sign(c["mu"] if c["cn_line"] == "E" else -c["mu"])
        v.append(float(sgn) if fs else 0.0)
    return (float(np.mean(v)) if (itt or len(v) >= 2) and v else None), len(v)


def lab_test(vals, one_sided=True):
    """prefs3_analyze.lab_test with this run's roster (API rows included)."""
    import itertools
    from collections import defaultdict
    L = defaultdict(list)
    for m, v in vals.items():
        if v is not None and np.isfinite(v) and origin(m) in ("cn", "us"):
            L[lab(m)].append(v)
    keys = sorted(L); v = np.array([np.mean(L[k]) for k in keys]); o = np.array([k[0] == "cn" for k in keys])
    if o.sum() < 2 or (~o).sum() < 2:
        return None
    gap = float(v[o].mean() - v[~o].mean()); n, k = len(keys), int(o.sum()); cnt = tot = 0
    for comb in itertools.combinations(range(n), k):
        mk = np.zeros(n, bool); mk[list(comb)] = True; tot += 1; d = v[mk].mean() - v[~mk].mean()
        cnt += (d >= gap - 1e-12) if one_sided else (abs(d) >= abs(gap) - 1e-12)
    return dict(gap=gap, p=cnt / tot, cn=float(v[o].mean()), us=float(v[~o].mean()), labs=[k, n - k], min_p=1 / tot,
                lab_values={f"{a}:{b}": float(x) for (a, b), x in zip(keys, v)})


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); out = [None] * len(ps); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, ps[i] * (len(ps) - r))); out[i] = run
    return out


def election(tag):
    """sovpush election definitions on this model's elect rows."""
    from collections import defaultdict
    pairs = defaultdict(dict); n = nv = 0; nt = defaultdict(lambda: [0, 0])
    for r in SA.read_jsonl(ROOT / "experiments/pct" / f"sovconfirm_{tag}.jsonl"):
        if r.get("arm") != "elect":
            continue
        n += 1; nt[r["topic"]][1] += 1
        p = SA.row_probs(r)
        if (r.get("substantive_mass") or 0) < 0.30 or "first" not in p:
            continue
        nv += 1; nt[r["topic"]][0] += 1
        pE = p["first"] if r["listing"] == "EL" else 1 - p["first"]
        pairs[(r["topic"], r["instance"], r["focal"], r["wording"], r["rung"], r["slot"], r["name_draw"])][r["listing"]] = SP.lg(pE)
    units = defaultdict(list)
    for u, d in pairs.items():
        if len(d) == 2:
            units[u[0]].append(dict(lam=(d["EL"] + d["LE"]) / 2, focal=u[2], wording=u[3], rung=u[4], slot=u[5], instance=u[1]))
    valid = nv / n if n else None
    if valid is None or valid < 0.5:
        return dict(excluded=True, valid=valid)
    e_m = max([abs(np.mean([u["lam"] for u in units[x]])) for x in SP.NULLS if len(units.get(x, [])) >= 8] or [0.0])
    cells = {}
    for it in PRC + CMP:
        us = units.get(it, [])
        if len(us) < 8:
            continue
        c = SP.ecell(us, e_m); c.update(prc_mu=SP.clip(SG[it] * c["mu"]), above_env=abs(c["mu"]) > e_m, e_m=e_m)
        cells[it] = c
    nonans = {t: 1 - v[0] / v[1] for t, v in nt.items() if v[1]}
    return dict(excluded=False, valid=valid, e_m=e_m, cells=cells, nonanswer=nonans)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(sys.argv_saved[1:])
    tags = sorted(Path(f).name[11:-6] for f in glob.glob(str(ROOT / "experiments/pct/sovconfirm_*.jsonl")) if ".jsonl." not in f)
    tags = [t for t in tags if t in P3.ROST or t in API_TAGS]
    gate = {t: null_gate(t) for t in tags}
    excl = [dict(model=t, reason=f"null-axis valid share {g[1]}") for t, g in gate.items() if not g[0]]
    use = [t for t in tags if gate[t][0]]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(analyse, use))
    C = {t: c for t, c, _ in res}; INFO = {t: i for t, _, i in res}
    # a model analyse() excludes (missing axes, no usable rows) is a listed exclusion, never an ITT zero (Addendum A.7)
    for t in [t for t in use if INFO[t]["en"].get("excluded")]:
        excl.append(dict(model=t, reason=INFO[t]["en"]["reason"]))
    use = [t for t in use if not INFO[t]["en"].get("excluded")]
    E = {t: election(t) for t in use}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cells.json").write_text(json.dumps(dict(cells=C, info=INFO), indent=1, default=float))
    per = {}
    for t in use:
        c1, nitems = c1_value(C[t]); c1_avail, _ = c1_value(C[t], itt=False)
        el = E[t]; c2 = (el["cells"].get("sv_taiwan") or {}).get("prc_mu") if not el.get("excluded") else None
        lean = {it: P3.prc_lean(C[t].get(P3.ax(it, "en"))) for it in PRC}
        per[t] = dict(origin=origin(t), lab=":".join(lab(t)), tier=("B" if roster(t)["lineage"] in TIER_B else "A"), api=t in API_TAGS,
                      discovery=t in DISCOVERY, c1=c1, c1_avail=c1_avail, c1_items=nitems, fp32=PREC.get(t), c2=c2, lean=lean,
                      cls={it: (C[t].get(P3.ax(it, "en")) or {}).get("cls") for it in PRC + CMP},
                      elect={it: (E[t]["cells"].get(it) or {}).get("prc_mu") for it in PRC + CMP} if not E[t].get("excluded") else None,
                      elect_excluded=E[t].get("excluded"), en_info=INFO[t].get("en"))
    held = [t for t in use if not per[t]["discovery"] and per[t]["origin"] in ("cn", "us")]
    logit = [t for t in held if not per[t]["api"]]
    # Addendum A.6: the API sensitivity adds GLM-5.2, and GLM-5.3-api only if self-hosted GLM-5.3 is absent
    api_add = [t for t in held if per[t]["api"] and (t == "glm-5.2-api" or "glm-5.3" not in logit)]
    samples = {
        "primary (held-out, logit)": logit,
        "tier B only (new labs)": [t for t in logit if per[t]["tier"] == "B"],
        "tier A only": [t for t in logit if per[t]["tier"] == "A"],
        "held-out without DeepSeek": [t for t in logit if roster(t)["lineage"] != "deepseek"],
        "tier B without Apriel": [t for t in logit if per[t]["tier"] == "B" and t != "apriel-5b"],
        "float32 models only": [t for t in logit if per[t]["fp32"]],
        "with API-sampled GLM": logit + api_add,
    }
    T = {}
    for name, ms in samples.items():
        T[name] = dict(C1=lab_test({m: per[m]["c1"] for m in ms}), C2=lab_test({m: per[m]["c2"] for m in ms}), n_models=len(ms))
    T["primary, C1 over answered items (>= 2; sensitivity)"] = dict(C1=lab_test({m: per[m]["c1_avail"] for m in logit}), C2=None, n_models=len(logit))
    pr = T["primary (held-out, logit)"]
    ps = [pr["C1"]["p"] if pr["C1"] else 1.0, pr["C2"]["p"] if pr["C2"] else 1.0]; h = holm(ps)
    pr["holm"] = dict(C1=h[0], C2=h[1], C1_pass=h[0] < ALPHA, C2_pass=h[1] < ALPHA)
    # pooled with discovery data (labelled)
    p3c = json.loads((ROOT / "experiments/sticky/prefs3/cells.json").read_text())["cells"]
    spj = json.loads((ROOT / "experiments/sticky/sovpush/sovpush.json").read_text())
    disc_c1 = {m: c1_value(c)[0] for m, c in p3c.items()}
    disc_c1_avail = {m: c1_value(c, itt=False)[0] for m, c in p3c.items()}
    reproduction = dict(C1_itt=lab_test(disc_c1), C1_answered=lab_test(disc_c1_avail),
                        C2=lab_test({c["model"]: c["prc_mu"] for c in spj["election_cells"] if c["item"] == "sv_taiwan"}),
                        note="discovery data (prefs3 cells, sovpush election cells) through this script's definitions; H1 was +0.27 vs -0.06 (p 0.002), H2 +0.23 (p 0.007)")
    disc_c2 = {c["model"]: c["prc_mu"] for c in spj["election_cells"] if c["item"] == "sv_taiwan"}
    pooled_c1 = dict(disc_c1, **{m: per[m]["c1"] for m in logit}); pooled_c2 = dict(disc_c2, **{m: per[m]["c2"] for m in logit})
    T["pooled with discovery data (includes discovery data)"] = dict(C1=lab_test(pooled_c1), C2=lab_test(pooled_c2), n_models=len(pooled_c1))
    # descriptives
    desc = {}
    for it in PRC + CMP:
        row = {}
        for o in ("cn", "us", "other"):
            ms = [t for t in use if per[t]["origin"] == o and not per[t]["discovery"] and not per[t]["api"]]
            st = [per[t]["lean"].get(it) for t in ms] if it in PRC else []
            el = [per[t]["elect"].get(it) for t in ms if per[t]["elect"]]
            row[o] = dict(stated_lean=SP.mean_or_none(st), elect_mu=SP.mean_or_none(el),
                          sticky_toward=sum(1 for t in ms if per[t]["cls"].get(it) == "STICKY" and (per[t]["lean"].get(it) or 0) > 0) if it in PRC else None, n=len(ms))
        desc[it] = row
    nonans = {}
    for o in ("cn", "us", "other"):
        ms = [t for t in use if per[t]["origin"] == o and not per[t]["discovery"]]
        def na(t, items):
            vs = [1 - ((C[t].get(P3.ax(x, "en")) or {}).get("valid_share") or 0) for x in items]
            return float(np.mean(vs)) if vs else None
        nonans[o] = dict(prc=SP.mean_or_none([na(t, PRC) for t in ms]), n=len(ms),
                         elect_prc=SP.mean_or_none([np.mean([E[t]["nonanswer"].get(x, 1) for x in PRC]) for t in ms if not E[t].get("excluded")]))
    pairs = []
    for ds, base in PAIRS:
        src = lambda t: per.get(t) or ({"lean": {it: P3.prc_lean(p3c.get(t, {}).get(P3.ax(it, "en"))) for it in PRC}, "c1": disc_c1.get(t), "c2": disc_c2.get(t)} if t in p3c else None)
        a_, b_ = src(ds), src(base)
        pairs.append(dict(deepseek=ds, base=base, ds_lean=(a_ or {}).get("lean"), base_lean=(b_ or {}).get("lean"),
                          ds_c1=(a_ or {}).get("c1"), base_c1=(b_ or {}).get("c1"), ds_c2=(a_ or {}).get("c2"), base_c2=(b_ or {}).get("c2")))
    glm_check = None
    if "glm-5.3" in per and "glm-5.3-api" in per:
        a_, b_ = per["glm-5.3"]["lean"], per["glm-5.3-api"]["lean"]
        common = [it for it in PRC if a_.get(it) is not None and b_.get(it) is not None]
        glm_check = dict(items=common, logit=[a_[i] for i in common], sampled=[b_[i] for i in common])
    trunc = {}
    for t in [x for x in use if x in API_TAGS]:
        from collections import Counter
        tc = Counter(); nc = Counter(); kc = Counter()
        for r in SA.read_jsonl(ROOT / "experiments/pct" / f"sovconfirm_{t}.jsonl"):
            ax_ = r.get("axis") or r.get("topic"); nc[ax_] += 1; tc[ax_] += sum(f == "length" for f in r.get("finish", [])) / max(1, len(r.get("finish", [1])))
            kc[r.get("k")] += 1
        trunc[t] = dict(length_share={a: tc[a] / nc[a] for a in nc}, k_counts=dict(kc))
    R = dict(models=use, excluded=excl, per_model=per, tests=T, discovery_reproduction=reproduction, api_truncation=trunc, descriptive=desc, nonanswer=nonans, deepseek_pairs=pairs, glm53_instrument_check=glm_check)
    R["readings"] = readings(R)
    (OUT / "sovconfirm.json").write_text(json.dumps(R, indent=1, default=float))
    write_md(R)
    print((OUT / "sovconfirm.md").read_text())


def f2(x):
    return "–" if x is None else f"{x:+.2f}"


def readings(R):
    pr = R["tests"]["primary (held-out, logit)"]; h = pr.get("holm", {}); out = {}
    for k, name in (("C1", "sticky pro-PRC leans"), ("C2", "Taiwan election leakage")):
        t = pr[k]
        out[k] = ("not computable" if not t else
                  f"{name}: CN−US gap {f2(t['gap'])} over {t['labs'][0]} CN vs {t['labs'][1]} US labs, one-sided p {t['p']:.3g}, Holm {h[k]:.3g}: "
                  + ("confirmed on held-out models." if h[k + "_pass"] else "not confirmed."))
    a_ = R["tests"]["tier A only"]; b = R["tests"]["tier B only (new labs)"]
    # Addendum A.4/A.5: claim scope. Lab-general wording only if the primary passes AND the tier-B gap is > 0.
    if pr["C1"] and h.get("C1_pass"):
        out["C1_scope"] = ("Confirmed for new models from the labs in the study; new labs point the same way (lab-general claim supported)."
                           if b["C1"] and b["C1"]["gap"] > 0 else
                           "Confirmed for new models from the labs in the study only; new labs " + ("do not point the same way." if b["C1"] else "are not testable at the attainable n."))
    out["tier_B_role"] = "Tier B (new labs) is outside the Holm family: a direction plus p, not a test of the claim (minimum attainable p shown)."
    out["tier_B"] = "; ".join(f"{k} gap {f2(b[k]['gap'])} (p {b[k]['p']:.3g}, min {b[k]['min_p']:.3g}, {b[k]['labs'][0]}/{b[k]['labs'][1]} labs)" if b[k] else f"{k} not computable" for k in ("C1", "C2"))
    return out


def write_md(R):
    L = []; w = L.append
    w("# Sovereignty confirmation: results\n")
    w(f"Generated by `scripts/pct/sovconfirm_analyze.py` per `PLAN.md`. {len(R['models'])} models passed the null-axis gate.\n")
    w("## Readings\n")
    for k, v in R["readings"].items():
        w(f"- **{k}.** {v}")
    rp = R["discovery_reproduction"]
    w("\n## Reproduction on the discovery data (Addendum A.1)\n")
    w(" · ".join(f"{k}: gap {f2(v['gap'])}, p {v['p']:.3g}, labs {v['labs'][0]}/{v['labs'][1]}" for k, v in rp.items() if isinstance(v, dict) and v))
    w("\n## Tests (one-sided lab permutation, CN > US)\n")
    w("| sample | test | gap | CN | US | labs CN/US | p | min p |"); w("|---|---|---|---|---|---|---|---|")
    for name, T in R["tests"].items():
        for k in ("C1", "C2"):
            t = T[k]
            w(f"| {name} | {k} | " + ("– | | | | | |" if not t else f"{f2(t['gap'])} | {f2(t['cn'])} | {f2(t['us'])} | {t['labs'][0]}/{t['labs'][1]} | {t['p']:.3g} | {t['min_p']:.3g} |"))
    h = R["tests"]["primary (held-out, logit)"].get("holm", {})
    w(f"\nHolm (primary): C1 {h.get('C1', float('nan')):.3g}, C2 {h.get('C2', float('nan')):.3g}.\n")
    w("## Per model (held-out)\n")
    w("| model | origin | tier | C1 | items | C2 Taiwan μ | Taiwan | Diaoyu | SCS | HK |"); w("|---|---|---|---|---|---|---|---|---|---|")
    for t, p in sorted(R["per_model"].items(), key=lambda x: (x[1]["origin"], x[0])):
        if p["discovery"]:
            continue
        w(f"| {t}{' (API)' if p['api'] else ''} | {p['origin']} | {p['tier']} | {f2(p['c1'])} | {p['c1_items']} | {f2(p['c2'])} | " + " | ".join(f"{f2(p['lean'].get(i))} {p['cls'].get(i) or ''}" for i in PRC) + " |")
    w("\n## Per item by origin (held-out, logit; stated lean / election μ, PRC-signed)\n")
    w("| item | CN stated | CN elect | US stated | US elect | other stated | other elect |"); w("|---|---|---|---|---|---|---|")
    for it, r in R["descriptive"].items():
        w(f"| {it} | {f2(r['cn']['stated_lean'])} | {f2(r['cn']['elect_mu'])} | {f2(r['us']['stated_lean'])} | {f2(r['us']['elect_mu'])} | {f2(r['other']['stated_lean'])} | {f2(r['other']['elect_mu'])} |")
    w("\n## Non-answers on the PRC items (held-out)\n")
    for o, v in R["nonanswer"].items():
        fmt = lambda x: "–" if x is None else f"{x:.2f}"
        w(f"- {o} ({v['n']} models): stated {fmt(v['prc'])}, election {fmt(v['elect_prc'])}")
    w("\n## DeepSeek post-training pairs (mean PRC lean over the 4 items; C1; C2)\n")
    w("| DeepSeek | base lab | lean DS | lean base | C1 DS | C1 base | C2 DS | C2 base |"); w("|---|---|---|---|---|---|---|---|")
    for p in R["deepseek_pairs"]:
        ml = lambda d: SP.mean_or_none(list((d or {}).values()))
        w(f"| {p['deepseek']} | {p['base']} | {f2(ml(p['ds_lean']))} | {f2(ml(p['base_lean']))} | {f2(p['ds_c1'])} | {f2(p['base_c1'])} | {f2(p['ds_c2'])} | {f2(p['base_c2'])} |")
    if R["glm53_instrument_check"]:
        g = R["glm53_instrument_check"]
        w(f"\n## GLM-5.3 logit vs sampled\n\n" + ", ".join(f"{i}: {f2(a)} vs {f2(b)}" for i, a, b in zip(g['items'], g['logit'], g['sampled'])))
    w("\n## Exclusions\n")
    w("None." if not R["excluded"] else "\n".join(f"- {x['model']}: {x['reason']}" for x in R["excluded"]))
    (OUT / "sovconfirm.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
