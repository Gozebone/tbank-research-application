"""Время и окна.

Yambda хранит timestamp как число 5-секундных интервалов от неизвестной эпохи.
Внутри проекта всё переводится в секунды (int64), чтобы окна задавались в
обычных единицах. Абсолютная дата не важна: сплиты и окна считаются от
относительных границ.
"""

from __future__ import annotations

SECONDS_PER_DAY = 86_400
YAMBDA_TS_UNIT_SECONDS = 5


def days_to_seconds(days: float) -> int:
    return int(round(days * SECONDS_PER_DAY))


def yambda_ts_to_seconds(ts):
    """Векторизованное/скалярное преобразование сырого timestamp Yambda в секунды."""
    return ts * YAMBDA_TS_UNIT_SECONDS
