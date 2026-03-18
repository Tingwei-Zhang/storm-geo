"""Shared GEO (Generative Engine Optimization) module for geo_examples and ugc_injections."""

from .geo_generator import (
    COMMON_SYSTEM_PROMPT,
    COMMON_USER_PROMPT_START,
    OUTPUT_ONLY_SOURCE,
    GEOGenerator,
    GEOMethod,
    GEORequest,
    GoalType,
    InjectionPosition,
    ModelCaller,
    apply_geo_chain,
    call_gpt,
)

__all__ = [
    "COMMON_SYSTEM_PROMPT",
    "COMMON_USER_PROMPT_START",
    "OUTPUT_ONLY_SOURCE",
    "GEOGenerator",
    "GEOMethod",
    "GEORequest",
    "GoalType",
    "InjectionPosition",
    "ModelCaller",
    "apply_geo_chain",
    "call_gpt",
]
