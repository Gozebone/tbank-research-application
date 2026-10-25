"""CLI: таблицы и график по результатам LLM-агента.

Сопоставляет пул, предложенный агентом, с сеткой и с итоговым отбором:
сколько признаков агент предложил, сколько прошло скрининг, сколько было
новых относительно сетки и сколько реально попало в модель.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.auto_features.generator import DEFAULT_EVENTS, DEFAULT_WINDOWS
from recsys_auto.config import load_config, resolve
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("agent_report")


def _in_grid(entity: str, metric: str, event: str | None, window) -> bool:
    if entity is None or metric is None:
        return False
    return event in DEFAULT_EVENTS and window in DEFAULT_WINDOWS


def build(cfg, agent_tag: str, train_tag: str, add_tag: str | None = None) -> None:
    base = resolve(cfg.paths.experiments) / "auto_search"
    pool = load_json(base / f"agent_pool_{agent_tag}.json")
    transcript = (
        load_json(base / f"agent_transcript_{agent_tag}.json")
        if (base / f"agent_transcript_{agent_tag}.json").exists()
        else {}
    )
    selection_path = base / f"selected_specs_{train_tag}.json"
    selection = load_json(selection_path) if selection_path.exists() else {}
    selected = {s["name"] for s in selection.get("base_specs", [])}

    add_selected: set[str] = set()
    if add_tag:
        add_path = base / f"selected_specs_{add_tag}.json"
        if add_path.exists():
            add_selected = {s["name"] for s in load_json(add_path).get("base_specs", [])}

    screen_path = base / f"screen_{train_tag}.csv"
    screen = pd.read_csv(screen_path) if screen_path.exists() else None
    grid_screen = pd.read_csv(base / "screen.csv") if (base / "screen.csv").exists() else None
    grid_names = set(grid_screen["name"]) if grid_screen is not None else set()

    rows = []
    for item in pool.get("specs", []):
        name = item["name"]
        row = {
            "name": name,
            "entity": item["entity"],
            "metric": item["metric"],
            "event_type": item.get("event_type"),
            "window_days": item.get("window_days"),
            "agent_user_auc": item.get("user_auc"),
            "agent_nan_rate": item.get("nan_rate"),
            "in_fixed_grid": _in_grid(
                item["entity"], item["metric"], item.get("event_type"), item.get("window_days")
            ),
            "selected_in_final_model": name in selected,
            "in_additive_model": name in add_selected,
            "hypothesis": item.get("hypothesis"),
        }
        if screen is not None and name in set(screen["name"]):
            hit = screen[screen["name"] == name].iloc[0]
            row["full_screen_user_auc"] = float(hit["user_auc"])
            row["full_screen_rank"] = int(screen.index[screen["name"] == name][0]) + 1
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("agent_user_auc", ascending=False)
    tables_dir = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables_dir / "agent_features.csv", index=False)

    stats = transcript.get("stats", {})
    summary = {
        "agent_tag": agent_tag,
        "train_tag": train_tag,
        "model": transcript.get("model"),
        "n_proposed_to_pool": len(pool.get("specs", [])),
        "n_new_vs_grid": int((~table["in_fixed_grid"]).sum()) if len(table) else 0,
        "n_new_vs_screen": int((~table["name"].isin(grid_names)).sum()) if len(table) else 0,
        "n_selected_in_final": int(table["selected_in_final_model"].sum()) if len(table) else 0,
        "n_selected_in_additive": int(table["in_additive_model"].sum()) if len(table) else 0,
        "n_extra_specs_added_to_screen": selection.get("n_extra_specs"),
        "n_selected_from_extra": selection.get("n_selected_from_extra"),
        "agent_stats": stats,
        "selected_features": table.loc[
            table["selected_in_final_model"], "name"
        ].tolist()
        if len(table)
        else [],
        "agent_summary": transcript.get("summary"),
        "screen_seconds": selection.get("screen_meta", {}).get("screen_seconds"),
    }
    save_json(summary, tables_dir / "agent_summary.json")
    log.info(
        "agent pool: %d specs, %d outside grid, %d selected in final model",
        summary["n_proposed_to_pool"],
        summary["n_new_vs_grid"],
        summary["n_selected_in_final"],
    )

    # график: эффект агента на val NDCG@10 (сетка / конкуренция / аддитивный)
    tags = [agent_tag, train_tag]
    if add_tag:
        tags.append(add_tag)
    metrics = {
        tag: load_json(base / f"metrics_{tag}.json")["results"]
        for tag in tags
        if (base / f"metrics_{tag}.json").exists()
    }
    variants = ["auto_base", "auto_rel", "manual_plus_auto"]
    variants = [v for v in variants if all(v in res for res in metrics.values())]
    if len(metrics) >= 2 and variants:
        rows = []
        for tag, res in metrics.items():
            for variant in variants:
                if variant not in res:
                    continue
                val = res[variant]["metrics"]["val"]
                test = res[variant]["metrics"].get("test", {})
                rows.append(
                    {
                        "tag": tag,
                        "variant": variant,
                        "n_features": res[variant]["n_features"],
                        "val_ndcg@10": round(val["ndcg@10"], 4),
                        "val_roc_auc": round(val["roc_auc"], 4),
                        "test_ndcg@10": round(test["ndcg@10"], 4) if test else None,
                    }
                )
        pd.DataFrame(rows).to_csv(tables_dir / "agent_variants.csv", index=False)

        labels = list(metrics)
        colors = ["#9aa5b1", "#2f6fd0", "#e08a1e"][: len(labels)]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        width = 0.8 / len(labels)
        xs = range(len(variants))
        all_values = []
        for index, (tag, res) in enumerate(metrics.items()):
            values = [res[v]["metrics"]["val"]["ndcg@10"] for v in variants]
            all_values.extend(values)
            offsets = [x - 0.4 + width * (index + 0.5) for x in xs]
            ax.bar(offsets, values, width, label=tag, color=colors[index])
            for offset, value in zip(offsets, values):
                ax.text(
                    offset,
                    value + 0.001,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        low, high = min(all_values), max(all_values)
        ax.set_ylim(low - 0.01, high + 0.018)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(variants)
        ax.set_ylabel("NDCG@10 (val)")
        ax.set_title("Эффект LLM-агента: grid / конкуренция / аддитивный набор", pad=12)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        figures = ensure_dir(resolve(cfg.paths.reports) / "figures")
        fig.savefig(figures / "agent_effect.png", dpi=150)
        plt.close(fig)
        log.info("figure written: agent_effect.png")
    else:
        log.warning("not enough metrics to build the agent comparison figure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-tag", default="quota")
    parser.add_argument("--train-tag", default="agent_quota")
    parser.add_argument("--add-tag", default="agent_add")
    args = parser.parse_args()
    cfg = load_config()
    build(cfg, args.agent_tag, args.train_tag, args.add_tag)


if __name__ == "__main__":
    main()
