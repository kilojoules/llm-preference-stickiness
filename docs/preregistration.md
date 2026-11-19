# Sovereignty confirmation: plan

Status: written 2026-11-18. The pods started before the review committee. This plan is frozen, with any committee addendum, before any sovconfirm row is opened. The user approved the run on 2026-11-18, with a $500 budget shared with the status-quo run.

**What is being confirmed.** Both observations come from data already seen, which is why this run exists.
- **H1, sticky lean.** In prefs3 (English), Chinese-origin labs hold FIRM-STICKY leans toward the PRC line on the 4 PRC items far more often than US-origin labs: net share +0.27 vs −0.06, lab permutation p 0.002 (post hoc). A FIRM-STICKY lean keeps its direction across rewording, re-ordering, format and forcing.
- **H2, Taiwan election leakage.** In sovpush, a candidate's Taiwan position moves CN-origin models' tie-break recommendations toward unification more than US-origin models': gap +0.23, p 0.007 (post hoc, 1 of 4 items).

## Models (none has seen any sovereignty battery)

Roster entries are in `scripts/sweep_models.json`; pods scA–scE, scL, scH1–scH3, scG.

**Tier B, new labs** (lineages new to the study):

| CN | US | other (reported, not tested) |
|---|---|---|
| Baichuan2-13B-Chat (baichuan) | rnj-1-instruct (Essential AI) | Solar-Open-100B (Upstage, KR) |
| Orion-14B-Chat (OrionStar) | Zamba2-7B-Instruct (Zyphra) | |
| TeleChat-12B (China Telecom) | Apriel-5B-Instruct (ServiceNow) | |
| LongCat-Flash-Lite (Meituan) | | |
| Step-3.5-Flash (StepFun) | | |
| MiniMax-M2.5 (MiniMax) | | |

**Tier A, unseen models from labs already in the study:**
- **CN:** qwen3-14b, qwen2.5-14b-instruct, qwen3.5-27b, qwen2.5-72b, glm-4.5-air, GLM-5.3 (self-hosted, FP8), deepseek-7b, DeepSeek-V4-Flash, hunyuan-a13b.
- **US:** gemma-4-12B, gemma-4-31B, llama-3.3-70b, gpt-oss-120b, phi-4, olmo-3-7b, nemotron-3-super-120b.

**DeepSeek post-training transfer (origin "mixed"; descriptive, not in C1/C2).** Each pair puts DeepSeek's R1 post-training on another lab's base, against that lab's own post-training of the same base. There are 4 pairs, so the read is a direction only.

| DeepSeek post-training | own lab's post-training |
|---|---|
| r1-distill-llama-8b | llama-3.1-8b-instruct (prefs3 panel) |
| r1-distill-llama-70b | llama-3.3-70b |
| r1-0528-qwen3-8b | qwen3-8b (panel) |
| r1-distill-qwen-14b | qwen2.5-14b-instruct |

**GLM-5.2 and GLM-5.3 via the z.ai API (sampled instrument).** z.ai returns no logprobs, so `scripts/pct/api_sampled.py` renders the identical prompts and samples each one:
- one sample per prompt at temperature 1.0 (the first ~70 prompts per model got 3; each row records its k). The z.ai coding endpoint sustains about 0.8 calls/s per model, so k > 1 would take days;
- 5.2 runs with thinking disabled; 5.3 with thinking "low", because it cannot be disabled;
- each sample is parsed to an option, and the probabilities are smoothed counts.

These rows enter C1/C2 only in a labelled sensitivity ("with API-sampled GLM"), never in the primary. GLM-5.3 run both ways, self-hosted logits and sampled, gives a direct check of the sampled instrument.

**Numerics.**
- Models up to ~45B run in float32, as prefs3 did.
- Larger models run in bf16 with the output head in float32: everything on the scL, scH and scG pods.
- Numerics are reported per model.

**Inclusion gate (mechanical, computed on null axes only, before the sovereignty cells are read).**
- A model enters if it produced rows.
- It must also have, in at least one format, a mean valid share ≥ 0.5 on the 8 English inline nulls.
- Within an included model, the prefs3 per-format gate (format valid share ≥ 0.5) and the inclusion rule of `prefs3_analyze.py` apply unchanged, with permissive gates as for prefs3's new-lab models.
- Every exclusion is listed with its reason.

## Battery

Built by `scripts/pct/sovconfirm_build.py`: 13,616 manifest rows plus 1,176 LIK grid prompts.
- **Stickiness:** the prefs3 English battery, verbatim, for the 4 PRC items, Crimea, Kosovo and the 8 English inline nulls. That is every arm (core, EXIT, ladder) in YN, PW and LIK.
- **Election:** the sovpush election arm, verbatim (6 items, 2 nulls, 3,072 rows).
- **Status quo:** the pre-review (v1) status-quo items. They are not analysed, because the items were revised before the 32-model run (`statusquo/PLAN.md`).

## Tests

Both are one-sided exact permutations of origin over labs (origin, lineage), predicted CN > US. A lab's value is the mean of its held-out models. Holm correction is applied over C1 and C2, and a test passes at Holm p < 0.05.

**C1, H1.** Per model, the net FIRM-STICKY-toward-PRC share over the 4 PRC items:
- per item: +1 if FIRM-STICKY toward the PRC line, −1 if FIRM-STICKY away from it, 0 otherwise (included but not FIRM-STICKY);
- averaged over the included items, with at least 2 of the 4 required;
- the classification, `cn_line` signing and gates are those of `prefs3_analyze.py`.

**C2, H2.** Per model, the election leakage on Taiwan:
- μ (mean logit P(name the E-pole candidate) over listing-order pairs), signed toward unification and clipped at ±2.9;
- at least 8 units;
- a model is dropped if more than half its election prompts fall below 0.30 name mass.

**Samples.**
- **Primary:** all held-out CN/US models (Tier A + B).
- **Key secondary:** Tier B only, which is selection-free at the lab level; the minimum attainable p is reported.
- **Also reported:**
  - Tier A only;
  - leaving out DeepSeek;
  - with the API-sampled GLM rows;
  - pooled with the discovery data (prefs3/sovpush; labelled "includes discovery data").

## Descriptive (no tests)

- C1 per item, and the dose-free lean (prefs3 PRC lean, clipped) by origin.
- C2 on the other 3 PRC items:
  - per-origin signed μ;
  - the Crimea/Kosovo charged envelope next to the bland-null envelope;
  - stated-vs-election direction agreement per item.
- **Non-answer rates by origin:** the share of prompts below 0.30 substantive or name mass, on the PRC items against the model's other axes.
- **DeepSeek transfer pairs:** the per-item stated PRC lean and C1/C2 values for each pair.
- **GLM-5.3:** the self-hosted vs sampled correlation of per-axis leans.

All interpretive sentences are computed from the JSON.

## Addendum A: review committee (2026-11-18)

Written after the pods started and before any sovconfirm row was opened.
- **Reviewers:** an Opus reviewer. The Fable reviewer covered the status-quo plan. The GPT-6.1 Sol and GLM-5.3 reviewers could not be reached (model_not_found; this session is not routed through model-router).
- **Analyzer:** `scripts/pct/sovconfirm_analyze.py` implements this plan plus this addendum.
- **Freeze:** this commit freezes the plan.

1. **Reproduction gate.** Through the analyzer's definitions, the discovery data give the following, reproducing H1/H2. These are stored in the JSON as `discovery_reproduction`.

   | quantity | gap | CN | US | labs CN/US | p |
   |---|---|---|---|---|---|
   | C1, answered items | +0.35 | +0.28 | −0.06 | 10/10 | 0.0018 |
   | C1, intent-to-treat | +0.28 | | | 12/10 | 0.0027 |
   | C2 | +0.23 | | | 10/10 | 0.0069 |

2. **Format gate on the nulls.** The per-format drop (mean valid share < 0.5) is computed on the 8 English inline nulls, not on the sovereignty cards. Otherwise refusals on the PRC items could drop a format and make FIRM-STICKY easier to reach. The substantive stickiness pass is the 6 sovereignty axes only; status-quo and election rows are not in it.
3. **C1 primary is intent-to-treat.**
   - Every gated model scores every PRC item, and an unanswered item counts 0 (denominator 4).
   - The answered-items rule (≥ 2 included items) is a sensitivity.
   - Reason: refusing these items is itself CN-typical. In discovery, all 3 models the ≥ 2 rule dropped were Chinese.
4. **Tier B is outside the Holm family.**
   - It is reported as a direction plus p, with its minimum attainable p (1/84 at 6 vs 3 labs; untestable with 1 US lab).
   - Its power at full discovery effect is about 0.35, so a non-pass is "not separable at the attainable n", not a refutation.
   - Tier B is also reported without Apriel-5B, whose lab is partly Canadian.
5. **Claim scope.**
   - A pass of the primary confirms "new models from the labs in the study".
   - A lab-general wording ("Chinese labs") additionally requires the Tier B C1 gap to be > 0.
   - Both sentences are generated by the analyzer.
6. **API-sampled GLM.**
   - All kept API rows ran with a 400-token cap. Rows from an earlier, larger cap were deleted before the restart. The first ~70 rows per model have k = 3 and the rest k = 1; each row records k.
   - The API sensitivity adds GLM-5.2, and adds GLM-5.3-api only if self-hosted GLM-5.3 is absent, so 5.3 is not counted twice in the glm lab.
   - Truncation (`finish == "length"`) is reported per axis.
   - The GLM-5.3 logit-vs-sampled comparison is labelled confounded by thinking mode: "low" via the API, closed self-hosted.
7. **Load failures.** A model that fails to load or fails the null gate is a listed exclusion with no substitution. Tier A and Tier B are whatever survives.
8. **Precision.** The large models run in bf16 or FP8 while the small ones run in float32, and in Tier B the large models are all Chinese. C1 and C2 are therefore also reported on float32 models only.
9. **Minor points.**
   - The panel comparator "llama-3.1-8b-instruct" is the tag `llama8b`.
   - C2's +0.23 is the best of 4 items, so expect shrinkage.
   - The exact permutation counts ties as at least as extreme.
