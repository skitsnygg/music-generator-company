from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ContextSpec:
    name: str
    prompt: str
    tags: List[str]
    bpm_min: int
    bpm_max: int
    mood: str
    genre: str


_CONTEXTS: Dict[str, ContextSpec] = {
    "focus": ContextSpec(
        name="focus",
        prompt="Ambient, minimal, steady, non-distracting. Soft textures, gentle motion. No vocals.",
        tags=["ambient", "minimal", "calm", "study", "focus"],
        bpm_min=60,
        bpm_max=90,
        mood="focus",
        genre="ambient",
    ),
    "workout": ContextSpec(
        name="workout",
        prompt=(
            "Dark, driving, song-like workout track. Steady 4/4 kick and bassline, melodic hook, "
            "clear sections (intro/build/drop/breakdown/outro). Avoid glitch, stutter, random chops, "
            "harsh artifacts. No vocals."
        ),
        tags=["dark", "driving", "melodic", "techno", "electro", "workout"],
        bpm_min=126,
        bpm_max=135,
        mood="workout",
        genre="electronic",
    ),
    "sleep": ContextSpec(
        name="sleep",
        prompt="Very calm, slow, warm, sparse, soothing pads, long reverb tails. No vocals.",
        tags=["sleep", "calm", "slow", "soothing", "ambient"],
        bpm_min=40,
        bpm_max=70,
        mood="sleep",
        genre="ambient",
    ),
}


def get_context_spec(name: str) -> ContextSpec:
    n = (name or "").strip().lower() or "focus"
    return _CONTEXTS.get(n, _CONTEXTS["focus"])


def list_contexts() -> List[str]:
    return sorted(_CONTEXTS.keys())


def build_prompt(context: str, *, user_prompt: Optional[str] = None) -> str:
    spec = get_context_spec(context)
    if user_prompt and str(user_prompt).strip():
        return f"{spec.prompt}\nUser prompt: {user_prompt}".strip()
    return spec.prompt
