"""Sovereignty confirmation battery (experiments/sticky/sovconfirm/PLAN.md), for held-out models.

- prefs3 English stickiness battery, verbatim, restricted to the 4 PRC items, the 2 bloc-alignment comparisons (Crimea,
  Kosovo) and the 8 English inline nulls: manifest rows (YN/PW, all arms) + the LIK grid filtered to those axes;
- the sovpush election arm (6 items x 2 instances x 192, plus the fair_month / park_name nulls), verbatim;
- the status-quo battery (data/statusquo_manifest.jsonl), verbatim.
Writes data/sovconfirm_manifest.jsonl and data/sovconfirm_lik_grid.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"
META = json.loads((D / "prefs3_meta.json").read_text())["topics"]
TOPICS = ["sv_taiwan", "sv_diaoyu", "sv_south_china_sea", "sv_hk_security_law", "sv_crimea", "sv_kosovo"]
AXES = {a for a, m in META.items() if m["lang"] == "en" and (m["topic"] in TOPICS or m["group"] in ("null", "surface"))}
assert len(AXES) == 14, sorted(AXES)

man = [json.loads(l) for l in open(D / "prefs3_manifest.jsonl")]
man = [r for r in man if r["axis"] in AXES]
man += [r for r in map(json.loads, open(D / "sovpush_manifest.jsonl")) if r["arm"] == "elect"]
man += [json.loads(l) for l in open(D / "statusquo_manifest.jsonl")]
keys = [r["key"] for r in man]
assert len(keys) == len(set(keys)), "duplicate keys"
with open(D / "sovconfirm_manifest.jsonl", "w") as f:
    for r in man:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

G = json.loads((D / "prefs3_lik_grid.json").read_text())
G["description"] = "sovconfirm LIK grid: prefs3_lik_grid.json restricted to the English arms and the 14 sovconfirm axes (scripts/pct/sovconfirm_build.py)."
arms = []
for a in G["arms"]:
    if a["lang"] != "en":
        continue
    ax = [x for x in a["items"]["axes"] if x in AXES]
    if ax:
        arms.append(dict(a, items=dict(a["items"], axes=ax)))
G["arms"] = arms
if isinstance(G.get("decode"), dict) and "axes" in G["decode"]:
    G["decode"] = dict(G["decode"], axes=[x for x in G["decode"]["axes"] if x in AXES])
(D / "sovconfirm_lik_grid.json").write_text(json.dumps(G, indent=1, ensure_ascii=False) + "\n")
from collections import Counter
print("sovconfirm manifest:", len(man), "rows;", dict(Counter(r.get("arm") if r.get("arm") in ("elect", "statusquo") else "prefs3_en" for r in man)))
print("grid arms:", [(a["name"], len(a["items"]["axes"])) for a in arms])
