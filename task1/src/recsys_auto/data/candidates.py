"""Генерация кандидатов (позитивы + негативы) для ranking-задачи."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from .split import Snapshot


def _popular_pool(
    events: pl.DataFrame,
    cutoff: int,
    pool_size: int,
    min_popularity: int,
) -> np.ndarray:
    """Пул негативов: самые популярные объекты в истории до cutoff."""
    pops = (
        events.filter((pl.col("ts") < cutoff) & (pl.col("event_type") == "listen"))
        .group_by("item_id")
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") >= min_popularity)
        # tie-break по item_id, иначе порядок пула (и сэмплирование негативов)
        # зависит от порядка строк внутри DuckDB/Polars.
        .sort(["cnt", "item_id"], descending=[True, False])
        .head(pool_size)
    )
    if pops.height == 0:
        return np.array([], dtype=np.uint32)
    return pops.get_column("item_id").to_numpy()


def _sample_negatives(
    positives_by_user: dict[int, np.ndarray],
    pool: np.ndarray,
    seen_by_user: dict[int, np.ndarray],
    n_negatives: int,
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    """Для каждого пользователя сэмплировать негативы вне его позитивов.

    ``seen_by_user`` — объекты, которые нельзя предлагать в качестве негатива
    (прослушанные в горизонте). Сами по себе старые прослушивания допустимы:
    повторное прослушивание — валидный позитив, а его отсутствие — валидный
    негатив.
    """
    out: dict[int, np.ndarray] = {}
    for uid, pos in positives_by_user.items():
        banned = seen_by_user.get(uid, np.array([], dtype=np.uint32))
        banned_set = set(banned.tolist()) | set(pos.tolist())
        n = min(n_negatives, max(len(pool) - len(banned_set), 0))
        if n <= 0:
            out[uid] = np.array([], dtype=np.uint32)
            continue
        # Отбор без замены: перемешиваем пул и берём первые допустимые.
        chosen: list[int] = []
        tries = 0
        while len(chosen) < n and tries < 20:
            draw = rng.choice(pool, size=min(len(pool), (n - len(chosen)) * 4 + 32), replace=False)
            for item in draw:
                item = int(item)
                if item not in banned_set:
                    chosen.append(item)
                    banned_set.add(item)  # без дублей
                    if len(chosen) == n:
                        break
            tries += 1
        out[uid] = np.asarray(chosen, dtype=np.uint32)
    return out


def build_candidates(
    events: pl.DataFrame,
    snapshot: Snapshot,
    max_positives_per_user: int,
    negatives_per_positive: int,
    pool_size: int,
    min_popularity: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Сформировать таблицу кандидатов для одного снимка.

    Колонки: uid, item_id, label, split, snapshot.
    """
    rng = np.random.default_rng(seed)
    horizon = events.filter(
        (pl.col("ts") >= snapshot.label_start) & (pl.col("ts") < snapshot.label_end)
    )
    # позитивы: уникальные пары (uid, item_id) прослушиваний в горизонте
    pos_df = (
        horizon.filter(pl.col("event_type") == "listen")
        .select(["uid", "item_id"])
        .unique()
        .to_pandas()
    )
    if pos_df.empty:
        raise ValueError(f"no positives for snapshot {snapshot.name}")

    # ограничение числа позитивов на пользователя.
    # Сортируем до sample: ``unique()`` не гарантирует порядок строк, а от него
    # зависит random_state, то есть состав негативов при пересборке датасета.
    pos_df = pos_df.sort_values(["uid", "item_id"], kind="mergesort").reset_index(drop=True)
    pos_df = pos_df.sample(frac=1.0, random_state=seed)
    pos_df = pos_df.groupby("uid", group_keys=False).head(max_positives_per_user)

    positives_by_user = {
        int(uid): grp["item_id"].to_numpy(dtype=np.uint32)
        for uid, grp in pos_df.groupby("uid", sort=False)
    }

    # объекты, которые пользователь уже слушал в горизонте (нельзя как негативы)
    seen_by_user = {
        int(uid): grp["item_id"].to_numpy(dtype=np.uint32)
        for uid, grp in pos_df.groupby("uid", sort=False)
    }

    pool = _popular_pool(events, snapshot.cutoff, pool_size, min_popularity)
    if pool.size == 0:
        raise ValueError("empty negative pool")

    neg_counts = {
        uid: min(negatives_per_positive * len(pos), 2000)
        for uid, pos in positives_by_user.items()
    }
    # _sample_negatives принимает единое n; вызываем отдельно для групп с разным n
    negatives: dict[int, np.ndarray] = {}
    by_n: dict[int, list[int]] = {}
    for uid, n in neg_counts.items():
        by_n.setdefault(n, []).append(uid)
    for n, uids in by_n.items():
        sub = {uid: positives_by_user[uid] for uid in uids}
        negatives.update(
            _sample_negatives(sub, pool, seen_by_user, n, np.random.default_rng(seed + n))
        )

    rows = []
    for uid, pos in positives_by_user.items():
        for item in pos:
            rows.append((uid, int(item), 1))
        for item in negatives.get(uid, np.array([], dtype=np.uint32)):
            rows.append((uid, int(item), 0))

    cand = pd.DataFrame(rows, columns=["uid", "item_id", "label"])
    cand["snapshot"] = snapshot.name
    cand["split"] = snapshot.split
    cand["cutoff"] = snapshot.cutoff
    cand["label_start"] = snapshot.label_start
    cand["label_end"] = snapshot.label_end
    # страховка от утечки позитива в негативы
    dup = cand.duplicated(["uid", "item_id"], keep=False)
    if dup.any():
        key = cand.loc[dup, ["uid", "item_id"]].drop_duplicates()
        pos_keys = cand.loc[cand.label == 1, ["uid", "item_id"]].drop_duplicates()
        leaked = key.merge(pos_keys, on=["uid", "item_id"])
        if len(leaked):
            raise AssertionError(f"{len(leaked)} positive pairs leaked into negatives")
    return cand
