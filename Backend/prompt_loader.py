"""
prompt_loader.py — Centralized system prompt loader
────────────────────────────────────────────────────
Loads prompt text files from Backend/prompts/ and returns them as strings.
Uses @lru_cache so each file is read from disk only once per process lifetime.

Usage:
    from prompt_loader import load_prompt, load_prompt_template

    # Static prompt
    system = load_prompt("classifier_system")

    # Prompt with template variables
    system = load_prompt_template("cad_ifc_system", today="2026-05-19")
"""

from __future__ import annotations

import os
from functools import lru_cache

# Resolve prompts/ directory relative to this file (Backend/prompts)
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompts",
)


@lru_cache(maxsize=128)
def load_prompt(name: str) -> str:
    """
    Read Backend/prompts/{name}.txt and return its content as a stripped string.
    Cached — each file is read only once per process lifetime.
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_prompt_template(name: str, **kwargs) -> str:
    """
    Load a prompt file and apply .format_map() with the given kwargs.
    Raises KeyError if a required placeholder is missing.
    """
    template = load_prompt(name)
    return template.format_map(kwargs)
