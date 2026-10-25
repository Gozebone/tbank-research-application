"""CLI: аддитивный вариант — сеточный отбор + признаки агента.

В основном эксперименте (`--extra-pool`) признаки агента конкурируют с
сеточными за одни и те же квоты. Здесь наоборот: берётся готовый сеточный
отбор (тег ``quota``) и к нему добавляются top-k признаков агента по
полному скринингу. Это разделяет два вопроса:

1. признаки агента хуже сеточных (конкуренция их вытесняет);
2. признаки агента полезны как дополнение к уже выбранному набору.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.generator import is_valid_combination
from recsys_auto.config import load_config, resolve
from recsys_auto.features.dsl import FeatureSpec
from recsys_auto.utils.io import get_logger, load_json, save_json

log = get_logger("agent_additive")


def _spec_from_row(row: dict) -> FeatureSpec:
    def clean(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return value

    window = clean(row.get("window_days"))
    return FeatureSpec(
        entity=row["entity"],
        metric=row["metric"],
        window_days=int(window) if window is not None else None,
        event_type=clean(row.get("event_type")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-tag", default="quota")
    parser.add_argument("--agent-pool", default=None)
    parser.add_argument(
        "--screen", default="screen_agent_quota.csv", help="combined screen (grid + agent)"
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-tag", default="agent_add")
    args = parser.parse_args()

    cfg = load_config()
    base = resolve(cfg.paths.experiments) / "auto_search"
    agent_pool_path = Path(args.agent_pool) if args.agent_pool else base / "agent_pool_quota.json"
    screen_path = Path(args.screen) if args.screen else base / "screen_agent_quota.csv"
    if not screen_path.is_absolute():
        screen_path = base / screen_path
    if not screen_path.exists():
        raise FileNotFoundError(
            f"{screen_path} not found; run stage screen with --extra-pool first"
        )

    selection = load_json(base / f"selected_specs_{args.grid_tag}.json")
    base_names = {s["name"] for s in selection["base_specs"]}

    pool = load_json(agent_pool_path)["specs"]
    pool_names = {item["name"] for item in pool}
    screen = pd.read_csv(screen_path).set_index("name")
    candidates = [name for name in pool_names if name in screen.index and name not in base_names]
    ranked = (
        screen.loc[candidates]
        .assign(strength=lambda df: (df["user_auc"] - 0.5).abs())
        .sort_values("strength", ascending=False)
    )
    chosen = list(ranked.head(args.top_k).index)

    extra_specs = []
    for name in chosen:
        row = screen.loc[name]
        spec = _spec_from_row(row.to_dict())
        if spec.metric in ("last_ts", "first_ts"):
            continue
        if not is_valid_combination(spec.entity, spec.metric, spec.event_type):
            continue
        extra_specs.append(spec)

    payload = dict(selection)
    payload.update(
        {
            "base_specs": selection["base_specs"] + [s.describe() for s in extra_specs],
            "extra_pool": str(agent_pool_path),
            "n_extra_specs": len(extra_specs),
            "n_selected_from_extra": len(extra_specs),
            "mode": "additive",
            "grid_tag": args.grid_tag,
        }
    )
    out = base / f"selected_specs_{args.out_tag}.json"
    save_json(payload, out)
    log.info(
        "wrote %s: %d grid + %d agent features",
        out,
        len(selection["base_specs"]),
        len(extra_specs),
    )
    for spec in extra_specs[:10]:
        log.info("  + %s", spec.name)


if __name__ == "__main__":
    main()
