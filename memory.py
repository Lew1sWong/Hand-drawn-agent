"""
Three-tier memory system with priority-based eviction.

  L1  distilled_style  — compact style summary, always injected into planner
  L2  working[]        — ≤ L2_MAX MemoryEntry items, priority-managed
  L3  archive[]        — unlimited, LLM-compressed when large

Priority score (0–1) per entry:
    score = 0.35 × freshness(t) + 0.30 × freq_score + 0.35 × quality

  freshness : exponential decay, half-life = 24 h
  freq_score: log(access_count+1) / log(20), capped at 1.0
  quality   : derived from user rating (0–1); 0.5 = no rating yet

Eviction flow (triggered when L2 would exceed L2_MAX):
  1. Find lowest-priority unpinned entry in L2
  2. Move it to L3 (archive)
  3. When L3 ≥ L3_DISTILL_AT, set a flag — agent calls async distillation

Rating → quality mapping:
    5 → 1.0 | 4 → 0.8 | 3 → 0.6 | no-rating → 0.5 | 2 → 0.2 | 1 → 0.1
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def memory_path_for(user_id: str) -> Path:
    """Return the memory file path for a user, creating the directory if needed."""
    d = Path(os.environ.get("MEMORY_DIR", "./memories"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"user_memory_{user_id}.json"

logger = logging.getLogger(__name__)

L2_MAX        = 20   # max working-memory slots
L3_DISTILL_AT = 30   # archive size that triggers LLM distillation
_HALF_LIFE_H  = 24   # freshness half-life in hours
_DECAY_K      = math.log(2) / _HALF_LIFE_H   # λ for exp decay

_RATING_TO_QUALITY = {5: 1.0, 4: 0.8, 3: 0.6, 2: 0.2, 1: 0.1}


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    content:        str
    created_ts:     float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)
    access_count:   int   = 1
    quality:        float = 0.5   # 0–1
    pinned:         bool  = False

    # ── Priority ──────────────────────────────────────────────────────

    def priority(self, now: Optional[float] = None) -> float:
        if self.pinned:
            return float("inf")
        now = now or time.time()
        age_h     = (now - self.last_access_ts) / 3600
        freshness = math.exp(-_DECAY_K * age_h)
        freq      = min(math.log(self.access_count + 1) / math.log(20), 1.0)
        return 0.35 * freshness + 0.30 * freq + 0.35 * self.quality

    def touch(self) -> None:
        self.last_access_ts = time.time()
        self.access_count  += 1

    def set_quality_from_rating(self, rating: int) -> None:
        self.quality = _RATING_TO_QUALITY.get(rating, 0.5)

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "content":        self.content,
            "created_ts":     self.created_ts,
            "last_access_ts": self.last_access_ts,
            "access_count":   self.access_count,
            "quality":        self.quality,
            "pinned":         self.pinned,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            content        = d.get("content", ""),
            created_ts     = d.get("created_ts", time.time()),
            last_access_ts = d.get("last_access_ts", time.time()),
            access_count   = d.get("access_count", 1),
            quality        = d.get("quality", 0.5),
            pinned         = d.get("pinned", False),
        )

    @classmethod
    def from_prompt(cls, prompt: str, quality: float = 0.5) -> "MemoryEntry":
        return cls(content=prompt, quality=quality)


# ---------------------------------------------------------------------------
# UserMemory
# ---------------------------------------------------------------------------

class UserMemory:
    def __init__(self, path: Path | str = "user_memory.json") -> None:
        self._path = Path(path)

        # L1 — always-injected summary
        self.distilled_style: str       = ""

        # L2 — working memory (priority-managed)
        self.working: list[MemoryEntry] = []

        # L3 — archive (evicted entries, awaiting distillation)
        self.archive: list[MemoryEntry] = []

        # Misc
        self.preferred_resolution: dict = {"width": 1280, "height": 720}
        self.language: str              = "zh"
        self.conversation_count: int    = 0
        self.notes: str                 = ""

        # In-memory only — reference to the last added entry for rating updates
        self._last_entry: Optional[MemoryEntry] = None

        # In-memory only — tracks good-entry count at last distillation to avoid redundant LLM calls
        self._last_distill_good_count: int = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> "UserMemory":
        if not self._path.exists():
            return self
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))

            self.distilled_style      = raw.get("distilled_style", "")
            self.preferred_resolution = raw.get("preferred_resolution", {"width": 1280, "height": 720})
            self.language             = raw.get("language", "zh")
            self.conversation_count   = raw.get("conversation_count", 0)
            self.notes                = raw.get("notes", "")

            self.working = [MemoryEntry.from_dict(e) for e in raw.get("working", [])]
            self.archive = [MemoryEntry.from_dict(e) for e in raw.get("archive", [])]

            # Migrate old format: successful_prompts list → MemoryEntry with quality=0.5
            for p in raw.get("successful_prompts", []):
                if p and not any(e.content == p for e in self.working):
                    self.working.append(MemoryEntry.from_prompt(p, quality=0.5))

            # Migrate old ratings list → update quality on matching working entries
            for r in raw.get("ratings", []):
                prompt, rating = r.get("prompt", ""), r.get("rating", 0)
                for e in self.working:
                    if e.content == prompt:
                        e.set_quality_from_rating(rating)
                        break

            logger.debug("Memory loaded: L2=%d L3=%d", len(self.working), len(self.archive))
        except Exception:
            logger.warning("Could not load memory from %s — starting fresh", self._path, exc_info=True)
        return self

    def _save_sync(self) -> None:
        data = {
            "distilled_style":      self.distilled_style,
            "preferred_resolution": self.preferred_resolution,
            "language":             self.language,
            "conversation_count":   self.conversation_count,
            "notes":                self.notes,
            "working":              [e.to_dict() for e in self.working],
            "archive":              [e.to_dict() for e in self.archive],
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        """Synchronous save — safe to call from sync contexts (feishu_bot, ratings)."""
        self._save_sync()

    async def save_async(self) -> None:
        """Non-blocking save for async contexts (agent, executor) — offloads I/O to thread pool."""
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_sync)

    # ------------------------------------------------------------------
    # L2 management
    # ------------------------------------------------------------------

    def add_entry(self, content: str, quality: float = 0.5) -> MemoryEntry:
        """
        Add a new MemoryEntry to L2 working memory.
        If an entry with the same content already exists, touch it instead.
        Evicts to L3 if L2 is full.
        """
        # De-duplicate
        for e in self.working:
            if e.content == content:
                e.touch()
                self._last_entry = e
                self.save()
                return e

        # Evict if full
        if len(self.working) >= L2_MAX:
            self._evict_one()

        self.evict_bad_memories()
        entry = MemoryEntry.from_prompt(content, quality=quality)
        self.working.append(entry)
        self._last_entry = entry
        self.save()
        return entry

    def _evict_one(self) -> None:
        """Move the lowest-priority unpinned L2 entry to L3."""
        now = time.time()
        candidates = [e for e in self.working if not e.pinned]
        if not candidates:
            return
        victim = min(candidates, key=lambda e: e.priority(now))
        self.working.remove(victim)
        self.archive.append(victim)
        logger.info(
            "Evicted to L3: %.60s… (priority=%.3f)",
            victim.content, victim.priority(now),
        )

    # ------------------------------------------------------------------
    # Public mutation helpers
    # ------------------------------------------------------------------

    def record_success(self, enhanced_prompt: str) -> None:
        """Call after a successful generation."""
        self.add_entry(enhanced_prompt, quality=0.5)
        self.conversation_count += 1
        self.save()

    def record_rating(self, prompt: str, rating: int) -> None:
        """Update quality of the most recent entry (or matching entry) with user rating."""
        quality = _RATING_TO_QUALITY.get(rating, 0.5)

        # Prefer updating the last added entry
        if self._last_entry is not None:
            self._last_entry.set_quality_from_rating(rating)
            self._last_entry.touch()
            self.save()
            logger.info("Rating %d/5 applied to last entry (quality→%.2f)", rating, quality)
            return

        # Fallback: find matching content in L2
        for e in self.working:
            if e.content == prompt:
                e.set_quality_from_rating(rating)
                e.touch()
                self.save()
                return

    def update_distilled(self, distilled: str) -> None:
        """Store LLM-distilled style summary (L1)."""
        self.distilled_style = distilled
        self.save()
        logger.info("L1 distilled style updated: %s", distilled[:100])

    def pin(self, content_substr: str) -> int:
        """Pin all L2 entries whose content contains the given substring. Returns count pinned."""
        count = sum(1 for e in self.working if content_substr in e.content and not e.pinned)
        for e in self.working:
            if content_substr in e.content:
                e.pinned = True
        if count:
            self.save()
        return count

    def update(self, **kwargs) -> None:
        allowed = {"preferred_resolution", "language", "notes"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(self, k, v)
        self.save()

    # ------------------------------------------------------------------
    # Distillation triggers (checked by agent.py)
    # ------------------------------------------------------------------

    def should_distill_working(self) -> bool:
        """True when enough new high-quality entries exist to re-distill L1."""
        good = [e for e in self.working if e.quality >= 0.6]
        n = len(good)
        if n < 5 or n <= self._last_distill_good_count:
            return False
        return n % 5 == 0

    def should_distill_archive(self) -> bool:
        """True when L3 archive is large enough to compress."""
        return len(self.archive) >= L3_DISTILL_AT

    def evict_bad_memories(self) -> int:
        """Remove entries that were rated poorly and never revisited."""
        before = len(self.working)
        self.working = [
            e for e in self.working
            if not (e.quality < 0.2 and e.access_count <= 1 and not e.pinned)
        ]
        removed = before - len(self.working)
        if removed:
            self.save()
            logger.info("Evicted %d bad-quality memories from L2", removed)
        return removed

    def top_working(self, n: int = 5, query: str = "") -> list[MemoryEntry]:
        """Return top-N L2 entries by priority, boosted by keyword overlap with query."""
        now = time.time()

        if query:
            # Simple bigram tokenisation — works for both Chinese and English without jieba
            def _tokens(text: str) -> set[str]:
                t = text.lower()
                unigrams = set(t.split())
                bigrams  = {t[i:i+2] for i in range(len(t) - 1) if t[i:i+2].strip()}
                return unigrams | bigrams

            query_tok = _tokens(query)

            def _score(e: MemoryEntry) -> tuple:
                overlap   = len(query_tok & _tokens(e.content))
                has_match = overlap > 0
                return (has_match, e.priority(now))
        else:
            def _score(e: MemoryEntry) -> tuple:
                return (True, e.priority(now))

        return heapq.nlargest(n, self.working, key=_score)

    def promote_from_archive(self, entries: list[MemoryEntry]) -> None:
        """Move distilled entries back to L2, clearing archive."""
        self.archive.clear()
        for e in entries:
            if len(self.working) >= L2_MAX:
                self._evict_one()
            self.working.append(e)
        self.save()

    # ------------------------------------------------------------------
    # Planner context string (L1 + top L2)
    # ------------------------------------------------------------------

    def as_context_str(self, query: str = "") -> str:
        lines = ["[User Memory]"]

        if self.distilled_style:
            lines.append(f"- Distilled style (high-confidence): {self.distilled_style}")

        res = self.preferred_resolution
        lines.append(f"- Preferred resolution: {res.get('width', 1280)}×{res.get('height', 720)}")
        if self.language:
            lines.append(f"- Language: {self.language}")
        if self.notes:
            lines.append(f"- Notes: {self.notes}")
        if self.conversation_count:
            lines.append(f"- Total generations: {self.conversation_count}")

        top = self.top_working(5, query=query)
        if top:
            lines.append("- Relevant past memories (by priority + keyword match):")
            for e in top:
                lines.append(f"    [q={e.quality:.1f}] {e.content[:120]}")

        return "\n".join(lines)
