"""LLM-агент для автоматического предложения признаков в рамках DSL.

Агент не пишет SQL: он предлагает валидные спецификации
``(entity, metric, event_type, window_days)``, которые затем проверяются
существующим скринингом и попадают в общий пул отбора.
"""

from .runner import AgentRun, run_agent
from .schema import SpecCandidate, describe_dsl, parse_candidates
from .tools import AgentTools

__all__ = [
    "AgentRun",
    "AgentTools",
    "SpecCandidate",
    "describe_dsl",
    "parse_candidates",
    "run_agent",
]
