"""Secure-by-default feature flags for local-only capabilities."""

from __future__ import annotations

import os
from collections.abc import Mapping


_TRUE_VALUES = frozenset({"1", "true", "yes"})


def read_env_flag(name: str, *, environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(name, "")).strip().lower() in _TRUE_VALUES


ENABLE_LOCAL_IMAGE_UPLOAD = read_env_flag("PLAMO_ENABLE_IMAGE_UPLOAD")
ENABLE_LOCAL_LLM = read_env_flag("PLAMO_ENABLE_LOCAL_LLM")
