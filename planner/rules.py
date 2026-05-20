"""
Deterministic intent classifier — runs BEFORE the LLM call. First match wins.
Returns a RuleResult with confidence=1.0 when intent is unambiguous.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class VideoMode(Enum):
    IMAGE_TO_VIDEO = auto()
    TEXT_TO_VIDEO  = auto()
    MULTI_SHOT     = auto()
    FIGURINE       = auto()
    AUDIO_PORTRAIT = auto()


class AudioMode(Enum):
    SFX       = auto()   # ambient / background sound
    NARRATION = auto()   # voice-over / TTS
    MIXED     = auto()   # both SFX and narration requested simultaneously
    UNKNOWN   = auto()   # ambiguous — let LLM decide


@dataclass
class RuleResult:
    video_mode:  VideoMode | None = None
    audio_mode:  AudioMode        = AudioMode.UNKNOWN
    wants_audio: bool             = False   # True if ANY audio keyword matched
    n_shots:     int | None       = None
    confidence:  float            = 0.0     # 1.0 = certain, skip LLM


_SFX_KW = frozenset([
    "background sound", "ambient", "sfx", "sound effect",
    "环境音", "背景音", "氛围音", "音效",
    "引擎", "风声", "雨声", "鸟叫", "狗叫", "engine", "wind", "rain",
])
_NARRATION_KW = frozenset([
    "narrate", "voice over", "voiceover", "旁白", "朗读",
    "让他说", "让她说", "说出", "念出",
    # 诗歌/台词/角色发声
    "吟诗", "吟唱", "朗诵", "念诗", "台词", "对白", "recite", "chant",
])
_AUDIO_ANY_KW = _SFX_KW | _NARRATION_KW | frozenset([
    "声音", "配音", "音乐", "sound", "audio", "narration", "music",
])
_MULTI_SHOT_KW = frozenset(["多镜头", "分镜", "multi-shot", "multi shot", "多场景"])
_FIGURINE_KW   = frozenset(["手办", "figurine", "figure", "toy", "q版"])


def classify(text: str, has_image: bool, has_audio: bool) -> RuleResult:
    t = text.lower()
    result = RuleResult()

    # ── Audio intent ─────────────────────────────────────────────────
    has_sfx       = any(k in t for k in _SFX_KW)
    has_narration = any(k in t for k in _NARRATION_KW)
    if has_sfx and has_narration:
        result.audio_mode  = AudioMode.MIXED
        result.wants_audio = True
    elif has_sfx:
        result.audio_mode  = AudioMode.SFX
        result.wants_audio = True
    elif has_narration:
        result.audio_mode  = AudioMode.NARRATION
        result.wants_audio = True
    elif any(k in t for k in _AUDIO_ANY_KW):
        result.wants_audio = True  # ambiguous — wants some audio, default sfx later

    # ── Video mode (first match wins) ────────────────────────────────
    if has_image and has_audio:
        result.video_mode = VideoMode.AUDIO_PORTRAIT
        result.confidence = 1.0

    elif any(k in t for k in _MULTI_SHOT_KW) or (
        # A numbered-scene script (≥ 2 markers) is unambiguously multi-shot.
        # Matches 场景1/scene1 and 镜头1/shot1 (the format the conversation layer suggests).
        len(re.findall(r"场景\s*\d+|scene\s*\d+|镜头\s*\d+|shot\s*\d+", t)) >= 2
    ):
        scenes = re.findall(r"场景\s*\d+|scene\s*\d+", t)
        result.video_mode = VideoMode.MULTI_SHOT
        result.n_shots    = max(len(scenes), 2)
        result.confidence = 1.0

    elif has_image and any(k in t for k in _FIGURINE_KW):
        result.video_mode = VideoMode.FIGURINE
        result.confidence = 1.0

    elif has_image:
        result.video_mode = VideoMode.IMAGE_TO_VIDEO
        result.confidence = 1.0

    else:
        # No image + not multi-shot → always text_to_video (LLM not needed)
        result.video_mode = VideoMode.TEXT_TO_VIDEO
        result.confidence = 1.0

    return result
