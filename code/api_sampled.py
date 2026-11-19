"""Sampled-instrument run for API-only models (GLM-5.2 / GLM-5.3 via the z.ai coding endpoint, which returns no logprobs).

Renders exactly the prompts the pod runner (stated_logit.py) would score -- the LIK grid arms plus the manifest rows -- and
for each prompt draws K samples at temperature 1.0. Each sample is parsed to an option (the runner's decode parsers, plus a
first-occurrence match for multi-word variants such as candidate names). Per prompt the option probabilities are the
smoothed counts p_o = (c_o + 0.5) / (n_parsed + 0.5 n_opt); lp_options = log p_o; valid_mass = parsed share,
substantive_mass = substantive-parsed share. Rows carry path="api_sampled" and the same fields as the runner's rows, so the
existing analyzers read them unchanged. Resumable. Raw samples go to <out>.samples.jsonl.

Usage: python3 scripts/pct/api_sampled.py --model glm-5.3 --out experiments/pct/sovconfirm_glm-5.3-api.jsonl
       --manifest data/sovconfirm_manifest.jsonl --grid data/sovconfirm_lik_grid.json --items_file data/prefs3_items_lik.json"""
import argparse, json, math, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pct")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "taskbias"))
import stated_logit as S
import api_backend as B

URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
THINK = {"glm-5.2": {"type": "disabled"}, "glm-5.3": {"type": "low"}}   # 5.3 cannot disable thinking; "low" is the minimum
LIK_LABELS = ["strongly disagree", "disagree", "agree", "strongly agree"]


def first_match(text, options, variants):
    t = text.lower()[:400]; best = None
    for name, vs in zip(options, variants):
        for v in vs:
            v = v.strip().lower()
            if not v:
                continue
            m = re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", t)
            if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(v) > best[2])):
                best = (m.start(), name, len(v))
    return best[1] if best else None


def parse_row(p, text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if p["_kind"] == "lik":
        lines = p["_lines"]
        opts = [str(num) for num, _ in lines]
        vars_ = [[str(num), f"{num})"] + ([LIK_LABELS[lab]] if lab is not None else [p["_optout_text"]]) for num, lab in lines]
        o = S.parse_pw(text, opts, vars_) or first_match(text, opts, vars_)
        if o is None:
            # a label stated without its number ("Agree"): the longest label that starts the reply
            low = text.lower().lstrip("*_ ")
            for (num, lab) in sorted(lines, key=lambda x: -len(LIK_LABELS[x[1]]) if x[1] is not None else 0):
                if lab is not None and low.startswith(LIK_LABELS[lab]):
                    return str(num)
        return o
    o = S.parse_decode(p.get("format"), text, p["options"], p["variants"])
    if o is not None and o not in p["options"]:   # e.g. parse_yn's "undecided" on a row whose option is named otherwise
        o = next((n for n, vs in zip(p["options"], p["variants"]) if o in [v.strip().lower() for v in vs]), None)
    return o if o is not None else first_match(text, p["options"], p["variants"])


def readout(p, parsed):
    n = len(parsed); ok = [x for x in parsed if x is not None]
    if p["_kind"] == "lik":
        lines = p["_lines"]; pos = [str(num) for num, _ in lines]
        cnt = {k: ok.count(k) for k in pos}; z = len(ok) + 0.5 * len(pos)
        lp_options = [None] * 4; lp_opt = None
        for (num, lab) in lines:
            lp = math.log((cnt[str(num)] + 0.5) / z) if ok else math.log(1 / len(pos))
            if lab is None:
                lp_opt = lp
            else:
                lp_options[lab] = lp
        sub = sum(cnt[str(num)] for num, lab in lines if lab is not None)
        ps = [math.exp(v) for v in lp_options]; zz = sum(ps)
        return dict(lp_options=lp_options, valid_mass=len(ok) / n, substantive_mass=sub / n, p_agree=(ps[2] + ps[3]) / zz,
                    score=sum(s * q for s, q in zip(S.SCORES, ps)) / zz, p_optout=(math.exp(lp_opt) if lp_opt is not None else None),
                    argmax_pos=max(range(len(lines)), key=lambda i: cnt[pos[i]]) + 1)
    opts = p["options"]; subst = set(p.get("substantive") or opts)
    cnt = {o: ok.count(o) for o in opts}; z = len(ok) + 0.5 * len(opts)
    lp = [math.log((cnt[o] + 0.5) / z) if ok else math.log(1 / len(opts)) for o in opts]
    return dict(lp_options=lp, valid_mass=len(ok) / n, substantive_mass=sum(cnt[o] for o in opts if o in subst) / n,
                argmax_option=max(opts, key=lambda o: cnt[o]))


class Client:
    def __init__(self, model, conc):
        self.model, self.key = model, B._key("zai")
        self.gate = threading.Semaphore(conc); self.lock = threading.Lock(); self.pause_until = 0.0; self.n429 = 0

    def call(self, messages, max_tokens):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0, "thinking": THINK[self.model]}
        data = json.dumps(body).encode()
        for attempt in range(12):
            while time.time() < self.pause_until:
                time.sleep(0.5)
            with self.gate:
                try:
                    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                    d = json.load(urllib.request.urlopen(req, timeout=300))
                    c = d["choices"][0]
                    u = d.get("usage") or {}
                    return (c["message"].get("content") or ""), (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0), c.get("finish_reason")
                except urllib.error.HTTPError as e:
                    msg = e.read()[:300].decode("utf8", "replace")
                    if e.code == 429 and "1302" in msg:   # request-rate limit: back off globally
                        with self.lock:
                            self.n429 += 1; self.pause_until = max(self.pause_until, time.time() + min(15, 2 + 2 * attempt))
                        continue
                    if e.code == 429:   # quota / balance: stop the run, do not burn retries
                        raise SystemExit(f"quota error: {msg}")
                    time.sleep(min(60, 3 * 2 ** attempt))
                except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError):
                    time.sleep(min(60, 3 * 2 ** attempt))
        return None, 0, "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(THINK))
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--items_file", default=None)
    ap.add_argument("--anchors", default=str(ROOT / "data/orderforce_anchors.json"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    prompts = []
    if a.grid:
        grid = json.loads(Path(a.grid).read_text())
        items = S.load_items_file(a.items_file) + S.load_extra_items(a.anchors)
        for p in S.build_prompts_grid(items, grid, None):
            p["_kind"] = "lik"; p["_lines"] = S.option_lines(p["order"], p["numbering"], None if p["optout"] == "none" else p["optout"], grid.get("optout_text", S.OPTOUT_TEXT))
            p["_optout_text"] = grid.get("optout_text", S.OPTOUT_TEXT); p["format"] = "LIK"
            prompts.append(p)
    man, _ = S.load_manifest(a.manifest, None)
    raw = {json.loads(l)["key"]: json.loads(l) for l in open(a.manifest)}
    for r in man:
        r["_kind"] = "manifest"; r["variants"] = raw[r["key"]]["variants"]; r["user"] = raw[r["key"]]["user"]; r["system"] = raw[r["key"]].get("system")
        prompts.append(r)
    out = Path(a.out); done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out) if l.strip()}
    todo = [p for p in prompts if p["key"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(prompts)} prompts ({sum(p['_kind'] == 'lik' for p in prompts)} LIK), {len(done)} done, {len(todo)} to do, k={a.k} -> {len(todo) * a.k} calls", flush=True)
    if a.dry:
        return
    cl = Client(a.model, a.conc); wl = threading.Lock(); t0 = time.time(); n_done = [0]
    sf = open(str(out) + ".samples.jsonl", "a"); of = open(out, "a")
    max_tok = 400   # the answer label comes first; a cap only truncates refusals and long reasoning (counted as non-answers)

    def work(p):
        msgs = ([{"role": "system", "content": p["system"]}] if p.get("system") else []) + [{"role": "user", "content": p["user"]}]
        res = [cl.call(msgs, max_tok) for _ in range(a.k)]
        texts = [r[0] for r in res]
        if any(t is None for t in texts):
            return
        parsed = [parse_row(p, t) for t in texts]
        row = {k: v for k, v in p.items() if not k.startswith("_") and k not in ("user", "system", "variants", "prefix_ids", "merged")}
        row.update(readout(p, parsed), model=f"zai/{a.model}", tag=f"{a.model}-api", steer=None, path="api_sampled", k=a.k,
                   reasoning_tokens=[r[1] for r in res], finish=[r[2] for r in res])
        with wl:
            sf.write(json.dumps(dict(key=p["key"], texts=texts, parsed=parsed), ensure_ascii=False) + "\n")
            of.write(json.dumps(row, ensure_ascii=False) + "\n"); of.flush(); sf.flush()
            n_done[0] += 1
            if n_done[0] % 200 == 0:
                el = time.time() - t0
                print(f"{n_done[0]}/{len(todo)} in {el / 60:.1f} min ({n_done[0] * a.k / el:.2f} calls/s; 429 backoffs {cl.n429})", flush=True)

    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(work, todo))
    print("API_DONE", n_done[0], flush=True)


if __name__ == "__main__":
    main()
