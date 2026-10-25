"""Скачивание датасетов.

Yambda раздаётся через HuggingFace Hub, KION — через GitHub Releases-подобные
raw-ссылки. Функции идемпотентны: уже скачанный файл нужного размера не
перекачивается.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from ..config import Config, resolve
from ..utils.io import ensure_dir, get_logger

log = get_logger(__name__)

YAMBDA_BASE = "https://huggingface.co/datasets/yandex/yambda/resolve/main"
KION_BASE = "https://github.com/irsafilo/KION_DATASET/raw/main"

YAMBDA_FILES = {
    "flat/500m/multi_event.parquet": 3_843_474_075,
    "artist_item_mapping.parquet": 38_274_621,
    "album_item_mapping.parquet": 54_528_894,
}

KION_FILES = {
    "interactions.csv.zip": 18_651_415,
    "data_en.zip": 77_308_689,
    "data_original.zip": 78_795_295,
    "sample_submission.csv": 14_094_920,
}


def download_file(url: str, dest: str | Path, expected_size: int | None = None) -> Path:
    """Скачать url в dest с докачкой и проверкой размера."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        log.info("skip (exists): %s", dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "recsys-auto-features/0.1"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if downloaded else "wb"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, mode) as fh:
        total = expected_size
        if total is None:
            length = resp.headers.get("Content-Length")
            total = int(length) + downloaded if length else None
        with tqdm(
            total=total,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            while True:
                chunk = resp.read(8 * 1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))

    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch for {dest}: {tmp.stat().st_size} != {expected_size}"
        )
    tmp.replace(dest)
    log.info("downloaded: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def download_yambda(cfg: Config) -> dict[str, Path]:
    raw = resolve(cfg.paths.raw) / "yambda"
    out: dict[str, Path] = {}
    for rel, size in YAMBDA_FILES.items():
        out[rel] = download_file(f"{YAMBDA_BASE}/{rel}", raw / rel, size)
    return out


def download_kion(cfg: Config) -> dict[str, Path]:
    raw = resolve(cfg.paths.raw) / "kion"
    out: dict[str, Path] = {}
    for rel, size in KION_FILES.items():
        out[rel] = download_file(f"{KION_BASE}/{rel}", raw / rel, size)
    return out


def extract_kion(cfg: Config) -> None:
    raw = resolve(cfg.paths.raw) / "kion"
    for archive in raw.glob("*.zip"):
        target = raw / archive.stem
        if target.exists() and any(target.iterdir()):
            log.info("skip extraction (exists): %s", target)
            continue
        ensure_dir(target)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        log.info("extracted %s -> %s", archive.name, target)


def extract_all(cfg: Config) -> None:
    extract_kion(cfg)


def free_space_gb(path: str | Path) -> float:
    return shutil.disk_usage(ensure_dir(path)).free / 1e9
