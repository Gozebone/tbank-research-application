"""Промпты LLM-агента.

Промпт намеренно узкий: агент не пишет SQL и не рассуждает о данных в
свободной форме, он ищет спецификации DSL, дополняющие уже отобранные
признаки.
"""

from __future__ import annotations

from .schema import describe_dsl_text

SYSTEM_PROMPT = """\
You are a feature-engineering researcher for a music recommender system.
Task: next-week listening prediction. For each user we rank candidate tracks
(true listens in the next 7 days vs. sampled popular negatives).
The stronger the feature separates positives inside one user's candidate list,
the better. Global AUC is not the target: within-user ranking is.

You do NOT write SQL and you have no access to raw data. You can only propose
features as DSL specifications:

    (entity, metric, event_type, window_days)

The harness computes them with SQL over the event log (strictly before the
snapshot cutoff, so no leakage) and screens them on a user sample using
within-user AUC.

{dsl}

HOW TO WORK
1. Call get_context to see the strongest grid features, model importances and
   what has already been accepted.
2. Propose {batch_size} candidates per call via propose_specs. Each candidate
   needs a one-sentence hypothesis explaining the mechanism.
3. Read the feedback: user_auc, auc, nan_rate, accepted/rejected. Weak
   candidates (|user_auc - 0.5| below the threshold) are rejected.
4. Iterate. Aim for COMPLEMENTARY features, not copies of the grid top:
   - combinations the fixed grid does not cover (unusual windows such as 2, 5,
     10, 21, 45, 120 days; like-only aggregates; organic-only aggregates);
   - entity/metric pairs that are underrepresented (see grid_coverage);
   - signals about breadth of taste, repetition, skips (mean_played),
     popularity of the item among other users (item/nunique_users),
     artist and album depth.
5. When you have enough accepted features (or the budget is nearly spent),
   call finish with a short summary.

HARD RULES
- entity/metric must be a valid pair (see the table above); played_ratio
  metrics and nunique_* only make sense for event_type=listen.
- last_ts / first_ts are NOT model features: recency and tenure are derived
  from them automatically. Do not propose them.
- At most {max_per_pair} accepted features per (entity, metric) pair: the
  harness rejects the rest with a "diversity cap" error. Spread proposals
  across entities and metrics instead of adding more windows of the same pair.
- Do not propose a spec that already exists in accepted_features.
- Do not repeat a rejected spec with identical parameters; change the
  mechanism (different entity, window, metric or event).
- One tool call at a time. Never answer with plain text instead of a tool call.
- All your reasoning outside tool arguments must be at most two sentences.
"""


def build_user_prompt(
    context: dict, budget_left: int, batch_size: int, extended: bool = False
) -> str:
    task = (
        "Start by proposing the first batch of complementary features. "
        "Include at least 3 binary expressions in the first call."
        if extended
        else "Start by proposing the first batch of complementary features."
    )
    return (
        "Initial context (already screened grid + current pool):\n"
        f"{_compact(context)}\n\n"
        f"Budget: {budget_left} propose_specs calls left, up to {batch_size} specs each.\n"
        f"{task}"
    )


def build_nudge(budget_left: int, batch_size: int) -> str:
    return (
        "You must respond with a tool call, not text. "
        f"Call propose_specs with up to {batch_size} new candidates "
        f"({budget_left} calls left), or finish if the pool is good enough."
    )


def _compact(payload: dict, max_chars: int = 6000) -> str:
    import json

    text = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


EXTENDED_BLOCK = """\
EXTENDED DSL (this run)
Besides plain aggregates you can build binary expressions over two aggregates:

    {op: ratio | delta | product, a: <aggregate>, b: <aggregate>}

- ratio   = a / |b|   -> share, intensity, trend (e.g. listens_7d / listens_30d)
- delta   = a - b     -> acceleration or drop between windows
- product = a * b     -> joint signal (item popularity x user loyalty)

Aggregates additionally support:
- condition: complete (played_ratio >= 0.9), skip (played_ratio <= 0.2),
  non_organic (is_organic = 0) — applied INSIDE the aggregate;
- metric decay_count with decay_days = half-life in days — time-decayed count,
  so recent events weigh more without choosing a hard window.

Recommended patterns (these are the gaps of the fixed grid):
- ratio(user_item.count.listen.7d, user_item.count.listen.30d) — interest trend
- delta(item.nunique_users.listen.7d, item.nunique_users.listen.30d)
- ratio(user_item.count.listen.complete.30d, user_item.count.listen.30d) — completion rate
- ratio(user_item.count.listen.skip.14d, user_item.count.listen.14d) — skip rate
- ratio(user_artist.decay_count.listen.30d.half_life=7, user_artist.count.listen.30d)
- product(item.nunique_users.listen.14d, user_artist.count.listen.90d)
- ratio(user_album.count.listen.30d, user.count.listen.30d) — album share

Rules for expressions:
- both sides must be valid aggregates; the result is computed on the same
  candidate rows, so there is no leakage;
- avoid using a metric that is zero for most candidates as the denominator of a
  ratio (the feature becomes noise); prefer count/decay_count as denominators;
- do not propose the same expression twice or reuse an accepted one;
- at most {max_per_pair} accepted expressions per aggregate pair;
- one hypothesis = one mechanism.

MANDATORY WORKFLOW
- Every propose call must contain at least 3 binary expressions
  (ratio/delta/product), not only plain aggregates.
- Plain aggregates are allowed as supporting proposals (conditions, decay,
  unusual windows), but they must not dominate the pool.
- Use propose_expressions for binary expressions and propose_aggregates for
  plain aggregates.

Example of a valid propose_expressions call:
{"candidates": [
  {"op": "ratio",
    "a": {"entity": "user_item", "metric": "count", "event_type": "listen",
           "window_days": 7, "condition": "none", "decay_days": 0},
    "b": {"entity": "user_item", "metric": "count", "event_type": "listen",
           "window_days": 30, "condition": "none", "decay_days": 0},
    "hypothesis": "interest accelerates when last week is a large share of the month"}
]}
"""


def system_prompt(batch_size: int, max_per_pair: int = 3, extended: bool = False) -> str:
    base = SYSTEM_PROMPT.format(
        dsl=describe_dsl_text(), batch_size=batch_size, max_per_pair=max_per_pair
    )
    if extended:
        return base + "\n" + EXTENDED_BLOCK.replace("{max_per_pair}", str(max_per_pair))
    return base
