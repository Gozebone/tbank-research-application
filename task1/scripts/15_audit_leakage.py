"""CLI: аудит утечек на реальных артефактах.

Проверки не декларативные, а по данным:

1. временные зазоры между train/val/test;
2. окна снимков (cutoff == label_start, горизонт 7 дней);
3. позитивы снимка совпадают с событиями в окне label;
4. ни один негатив не является позитивом того же пользователя;
5. признаки воспроизводятся независимым SQL только по `ts < cutoff`;
6. добавление событий из окна label меняет признак (значит, утечка была бы
   видна, и наши значения ей не равны);
7. отпечатки данных в метаданных совпадают с текущими событиями.

    python scripts/15_audit_leakage.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys_auto.config import load_config, resolve
from recsys_auto.utils.fingerprint import dataset_fingerprint
from recsys_auto.utils.io import ensure_dir, get_logger, load_json, save_json

log = get_logger("audit")

DAY = 86_400
ROWS: list[dict] = []


def record(check: str, ok: bool, detail: str) -> None:
    ROWS.append({"check": check, "status": "ok" if ok else "FAIL", "detail": detail})
    log.info("%-45s %s  %s", check, "ok" if ok else "FAIL", detail)


def audit(config_path: str | None) -> pd.DataFrame:
    cfg = load_config(config_path)
    split = load_json(resolve(cfg.paths.processed) / "split.json")
    datasets = resolve(cfg.paths.processed) / "datasets"
    events = str(resolve(cfg.prepare.events_file))

    # 1. зазоры
    gap = int(cfg.split.gap_days) * DAY
    ok = (
        split["train_end"] + gap == split["val_start"]
        and split["val_end"] + gap == split["test_start"]
    )
    record(
        "split gaps",
        ok,
        f"train_end={split['train_end']}, val_start={split['val_start']}, "
        f"test_start={split['test_start']}, gap={gap}",
    )

    # 2. окна снимков
    horizon = int(cfg.split.label_horizon_days) * DAY
    windows_ok = True
    details = []
    for name in ["train_w0", "train_w1", "train_w2", "train_w3", "val", "test"]:
        meta = load_json(datasets / f"{name}.meta.json")
        snap = meta.get("snapshot", meta)
        good = (
            snap["cutoff"] == snap["label_start"]
            and snap["label_end"] - snap["label_start"] == horizon
        )
        windows_ok &= good
        details.append(f"{name}:{snap['label_start']}..{snap['label_end']}")
    record("snapshot windows", windows_ok, ", ".join(details))

    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{events}')")

    for name in ["val", "test"]:
        cand = pd.read_parquet(
            datasets / f"{name}.parquet", columns=["uid", "item_id", "label", "cutoff"]
        )
        cutoff = int(cand["cutoff"].iloc[0])
        label_end = cutoff + horizon

        # 3. позитивы == события в окне label (на подвыборке пользователей)
        rng = np.random.default_rng(42)
        uids = rng.choice(cand["uid"].unique(), size=min(100, cand["uid"].nunique()), replace=False)
        positives = set(
            map(tuple, cand.loc[cand["label"] == 1, ["uid", "item_id"]].to_numpy().tolist())
        )
        events_window = con.execute(
            "SELECT uid, item_id FROM events WHERE event_type='listen' "
            f"AND ts >= {cutoff} AND ts < {label_end} AND uid IN ({','.join(map(str, uids))}) "
            "GROUP BY uid, item_id"
        ).fetch_df()
        expected = set(map(tuple, events_window.to_numpy().tolist()))
        got = {pair for pair in positives if pair[0] in set(uids)}
        record(
            f"{name}: positives == events in label window",
            got <= expected and len(got) > 0,
            f"{len(got)} positives, all present in events: {got <= expected}",
        )

        # 4. негатив не является позитивом
        dup = cand.groupby(["uid", "item_id"])["label"].nunique()
        record(
            f"{name}: no negative/positive duplicates",
            bool((dup <= 1).all()),
            f"rows={len(cand)}, duplicate pairs={(dup > 1).sum()}",
        )

    # 5-6. признаки воспроизводятся только по ts < cutoff
    val = pd.read_parquet(datasets / "val.parquet", columns=["uid", "cutoff"])
    cutoff = int(val["cutoff"].iloc[0])
    uids = np.random.default_rng(0).choice(val["uid"].unique(), size=300, replace=False)
    uid_list = ",".join(map(str, uids))
    correct = con.execute(
        "SELECT uid, count(*) FILTER (WHERE ts >= "
        f"{cutoff - 7 * DAY}) AS c7 FROM events "
        f"WHERE event_type='listen' AND ts < {cutoff} AND uid IN ({uid_list}) GROUP BY uid"
    ).fetch_df()
    leaky = con.execute(
        "SELECT uid, count(*) AS c7 FROM events "
        f"WHERE event_type='listen' AND ts >= {cutoff - 7 * DAY} "
        f"AND ts < {cutoff + horizon} AND uid IN ({uid_list}) GROUP BY uid"
    ).fetch_df()
    materialized = pd.read_parquet(
        resolve(cfg.paths.experiments) / "auto_search" / "features_quota" / "val.parquet",
        columns=["uid", "user__count__listen__7d"],
    )
    materialized = materialized[materialized["uid"].isin(uids)].drop_duplicates("uid")

    frame = pd.DataFrame({"uid": uids})
    frame = frame.merge(correct, on="uid", how="left").fillna({"c7": 0})
    frame = frame.merge(
        leaky.rename(columns={"c7": "leaky_c7"}), on="uid", how="left"
    ).fillna({"leaky_c7": 0})
    frame = frame.merge(
        materialized.rename(columns={"user__count__listen__7d": "materialized"}),
        on="uid",
        how="left",
    )
    match = (frame["c7"].astype(int) == frame["materialized"].astype(int)).mean()
    differs = (frame["c7"] != frame["leaky_c7"]).sum()
    record(
        "feature = SQL with ts < cutoff",
        bool(match == 1.0),
        f"{len(frame)} users, exact match {match:.0%}",
    )
    record(
        "future events would change the feature",
        bool(differs > 0),
        f"{differs}/{len(frame)} users differ if label window is included",
    )

    # 7. отпечатки
    expected_fp = dataset_fingerprint(cfg)
    fps = {
        name: load_json(datasets / f"{name}.meta.json").get("data_fingerprint")
        for name in ["val", "test", "train_w0"]
    }
    record(
        "dataset fingerprints",
        all(fp == expected_fp for fp in fps.values()),
        f"expected {expected_fp[:12]}..., got {[str(f)[:12] for f in fps.values()]}",
    )
    con.close()
    return pd.DataFrame(ROWS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    table = audit(args.config)
    tables = ensure_dir(resolve(cfg.paths.reports) / "tables")
    table.to_csv(tables / "leakage_audit.csv", index=False)
    save_json(
        {"n_checks": len(table), "n_failed": int((table["status"] != "ok").sum())},
        tables / "leakage_audit.json",
    )
    print(table.to_string(index=False))
    if (table["status"] != "ok").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
