"""CLI: LLM-агент, предлагающий признаки в рамках DSL.

Агент не пишет SQL: он предлагает ``(entity, metric, event_type, window_days)``,
харнесс валидирует их и скрининит на подвыборке пользователей, после чего
принятые признаки сохраняются в ``agent_pool_<tag>.json``.

Пул затем подмешивается к сеточным кандидатам:

    python scripts/06_auto_search.py --stage screen --tag agent_quota \\
        --per-entity 20 --extra-pool experiments/auto_search/agent_pool_quota.json
    python scripts/06_auto_search.py --stage materialize --tag agent_quota
    python scripts/06_auto_search.py --stage train --tag agent_quota --derived-top-k 50 --no-ablate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.agent import AgentTools, run_agent
from recsys_auto.config import load_config
from recsys_auto.utils.io import get_logger

log = get_logger("agent_cli")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--tag", default="quota")
    parser.add_argument("--iterations", type=int, default=None, help="лимит шагов диалога")
    parser.add_argument("--batch-size", type=int, default=None, help="кандидатов за вызов")
    parser.add_argument("--sample-users", type=int, default=None, help="пользователей для скрининга")
    parser.add_argument("--model", default=None, help="например qwen/qwen3.8-omni-flash")
    parser.add_argument("--max-propose-calls", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min-user-auc-gain", type=float, default=None)
    parser.add_argument("--price-in", type=float, default=None, help="$ за 1M входных токенов")
    parser.add_argument("--price-out", type=float, default=None, help="$ за 1M выходных токенов")
    parser.add_argument("--reset", action="store_true", help="начать пул с нуля")
    parser.add_argument(
        "--novel-only",
        action="store_true",
        help="отклонять признаки, уже покрытые фиксированной сеткой",
    )
    parser.add_argument(
        "--expr",
        action="store_true",
        help="расширенный DSL: условия, затухание и бинарные выражения",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tools = AgentTools(
        cfg,
        tag=args.tag,
        sample_users=args.sample_users,
        batch_size=args.batch_size,
        min_user_auc_gain=args.min_user_auc_gain,
        require_novel=args.novel_only,
        mode="expr" if args.expr else "flat",
    )
    if args.reset:
        tools.pool.clear()
        tools._save_pool()

    run = run_agent(
        cfg,
        tools,
        model=args.model,
        max_iterations=args.iterations,
        max_propose_calls=args.max_propose_calls,
        temperature=args.temperature,
        price_in=args.price_in,
        price_out=args.price_out,
        verbose=not args.quiet,
    )

    stats = run.stats()
    log.info(
        "agent done: pool=%d, propose_calls=%d, cost=$%.4f, finish=%s",
        len(run.pool),
        stats["n_propose_calls"],
        stats["cost_usd"],
        stats["finish_reason"],
    )
    accepted = sorted(
        run.pool.values(), key=lambda x: abs(x.get("user_auc", 0.5) - 0.5), reverse=True
    )
    for item in accepted:
        log.info(
            "  %-55s user_auc=%.3f nan=%.2f  %s",
            item["name"],
            item["user_auc"],
            item["nan_rate"],
            (item.get("hypothesis") or "")[:60],
        )
    print(f"\npool: {tools.pool_path}")
    print(f"transcript: {run.transcript_path}")
    print(f"cost: ${stats['cost_usd']:.4f}, tokens: {stats['prompt_tokens']}+{stats['completion_tokens']}")


if __name__ == "__main__":
    main()
