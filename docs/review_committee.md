# Sovconfirm: review committee (2026-11-18)

**Reviewers.**
- **Opus:** reviewed the plan, the roster/pods and the builders.
- **Fable:** reviewed the status-quo plan (see `../statusquo/review_committee.md`).
- **GPT-6.1 Sol and GLM-5.3:** both failed with model_not_found. This session is not routed through model-router; `/model-router:setup` should fix it.

**Opus findings.** All are adopted in PLAN.md Addendum A:
- write the analyzer before data and reproduce discovery;
- compute the format gate on the nulls;
- make C1 intent-to-treat;
- keep Tier B out of Holm, with power stated;
- scope the claim wording;
- fix the API-row instrument issues;
- state load-failure handling;
- add an fp32-only sensitivity;
- minor points.

**Roster check.** No mismatches: every planned model is in exactly one pod, the origin and lineage tags are consistent, and no held-out model is in any earlier sovereignty battery.
