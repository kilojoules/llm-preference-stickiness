"""OpenAI-compatible chat backend for API-only models (GLM-5.3 via z.ai). Key is read from
~/.config/model-router/secrets.toml (never printed). Thinking disabled to match the non-thinking
convention used for Qwen3. chat() returns (text, completion_tokens, finished)."""
import json, os, re, time, urllib.request, urllib.error

ENDPOINTS = {"glm-5.3": ("https://api.z.ai/api/coding/paas/v4/chat/completions", "zai", "glm-5.3")}


def _key(name):
    s = open(os.path.expanduser("~/.config/model-router/secrets.toml")).read()
    m = re.search(rf'^\s*{name}\s*=\s*"([^"]+)"', s, re.M)
    if not m:
        raise RuntimeError(f"no secret {name} in model-router secrets")
    return m.group(1)


def is_api_model(model_id):
    return model_id in ENDPOINTS


def chat(model_id, messages, max_tokens, temperature=0.7, top_p=0.95, retries=6, timeout=600):
    url, keyname, remote = ENDPOINTS[model_id]
    body = {"model": remote, "messages": messages, "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    if temperature > 0:
        body.update(temperature=temperature, top_p=top_p)
    else:
        body.update(temperature=0.0)
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key(keyname)}"})
            d = json.load(urllib.request.urlopen(req, timeout=timeout))
            c = d["choices"][0]
            return c["message"].get("content") or "", int(d.get("usage", {}).get("completion_tokens", 0)), c.get("finish_reason") == "stop"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, OSError) as e:
            last = e
            time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(f"api call failed after {retries} attempts: {last}")
