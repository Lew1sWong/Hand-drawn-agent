"""
Unit tests for the three-tier memory system.
No API calls required.
"""
import math
import time
import pytest
from memory import UserMemory, MemoryEntry, L2_MAX


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

class TestMemoryEntry:
    def test_from_prompt_defaults(self):
        e = MemoryEntry.from_prompt("hand-drawn village at dusk")
        assert e.content == "hand-drawn village at dusk"
        assert e.quality == 0.5
        assert e.access_count == 1
        assert e.pinned is False

    def test_priority_increases_with_quality(self):
        low  = MemoryEntry.from_prompt("test", quality=0.1)
        high = MemoryEntry.from_prompt("test", quality=1.0)
        assert high.priority() > low.priority()

    def test_priority_decays_over_time(self):
        old = MemoryEntry.from_prompt("old memory")
        old.last_access_ts -= 48 * 3600  # 48 hours ago
        new = MemoryEntry.from_prompt("new memory")
        assert new.priority() > old.priority()

    def test_pinned_entry_has_infinite_priority(self):
        e = MemoryEntry(content="must keep", pinned=True)
        assert math.isinf(e.priority())

    def test_touch_bumps_access_count(self):
        e = MemoryEntry.from_prompt("test")
        before = e.access_count
        e.touch()
        assert e.access_count == before + 1

    def test_touch_updates_last_access_ts(self):
        e = MemoryEntry.from_prompt("test")
        e.last_access_ts -= 100
        before = e.last_access_ts
        e.touch()
        assert e.last_access_ts > before

    def test_quality_rating_mapping(self):
        e = MemoryEntry.from_prompt("test")
        e.set_quality_from_rating(5)
        assert e.quality == 1.0
        e.set_quality_from_rating(1)
        assert e.quality == 0.1
        e.set_quality_from_rating(3)
        assert e.quality == 0.6

    def test_roundtrip_serialisation(self):
        e = MemoryEntry(content="test content", quality=0.8, pinned=True, access_count=5)
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.content      == e.content
        assert e2.quality      == e.quality
        assert e2.pinned       == e.pinned
        assert e2.access_count == e.access_count


# ---------------------------------------------------------------------------
# UserMemory — working memory management
# ---------------------------------------------------------------------------

class TestUserMemory:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_record_success_adds_entry(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.record_success("golden-hour village animation")
        assert len(mem.working) == 1
        assert "village" in mem.working[0].content

    def test_add_entry_with_quality(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("high quality prompt", quality=0.9)
        assert mem.working[0].quality == 0.9

    def test_duplicate_entry_is_touched_not_duplicated(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.record_success("same prompt")
        mem.record_success("same prompt")
        assert len(mem.working) == 1
        assert mem.working[0].access_count == 2

    def test_eviction_when_l2_full(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(L2_MAX):
            mem.add_entry(f"prompt number {i}", quality=0.5)
        assert len(mem.working) == L2_MAX
        # One more → lowest priority entry evicted to L3
        mem.add_entry("one more prompt")
        assert len(mem.working) == L2_MAX
        assert len(mem.archive) == 1

    def test_pinned_entry_never_evicted(self, tmp_path):
        mem = self._mem(tmp_path)
        pinned = MemoryEntry(content="PINNED", quality=0.1, pinned=True)
        mem.working.append(pinned)
        for i in range(L2_MAX):
            mem.add_entry(f"prompt {i}", quality=1.0)
        assert any(e.pinned for e in mem.working)

    def test_record_rating_updates_last_entry(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.record_success("test prompt")
        mem.record_rating("test prompt", 5)
        assert mem.working[0].quality == 1.0

    def test_record_rating_low_score(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.record_success("test prompt")
        mem.record_rating("test prompt", 1)
        assert mem.working[0].quality == 0.1

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = UserMemory(path).load()
        mem.record_success("persistent memory test")
        mem.save()

        mem2 = UserMemory(path).load()
        assert len(mem2.working) == 1
        assert "persistent" in mem2.working[0].content

    def test_quality_survives_reload(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = UserMemory(path).load()
        mem.add_entry("quality test", quality=0.8)
        mem.save()

        mem2 = UserMemory(path).load()
        assert mem2.working[0].quality == 0.8

    def test_distilled_style_roundtrip(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = UserMemory(path).load()
        mem.update_distilled("User loves golden-hour lighting.")
        mem.save()

        mem2 = UserMemory(path).load()
        assert "golden-hour" in mem2.distilled_style

    def test_as_context_str_includes_distilled(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.update_distilled("User loves hand-drawn style.")
        ctx = mem.as_context_str()
        assert "hand-drawn" in ctx

    def test_should_distill_working_false_when_few_entries(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.record_success("one entry")
        assert mem.should_distill_working() is False

    def test_should_distill_working_true_at_5_good_entries(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(5):
            mem.add_entry(f"good prompt {i}", quality=0.8)
        assert mem.should_distill_working() is True

    def test_top_working_returns_highest_priority(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("low quality",  quality=0.1)
        mem.add_entry("high quality", quality=1.0)
        top = mem.top_working(1)
        assert "high quality" in top[0].content

    def test_pin_prevents_eviction(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("important", quality=0.1)
        mem.pin("important")
        assert mem.working[0].pinned is True


# ---------------------------------------------------------------------------
# evict_bad_memories
# ---------------------------------------------------------------------------

class TestEvictBadMemories:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_removes_low_quality_single_access(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("bad prompt", quality=0.1)
        removed = mem.evict_bad_memories()
        assert removed == 1
        assert len(mem.working) == 0

    def test_keeps_low_quality_if_accessed_multiple_times(self, tmp_path):
        mem = self._mem(tmp_path)
        e = mem.add_entry("bad but accessed", quality=0.1)
        e.touch()  # access_count → 2
        removed = mem.evict_bad_memories()
        assert removed == 0
        assert len(mem.working) == 1

    def test_keeps_pinned_even_if_low_quality(self, tmp_path):
        mem = self._mem(tmp_path)
        e = mem.add_entry("pinned bad", quality=0.1)
        e.pinned = True
        removed = mem.evict_bad_memories()
        assert removed == 0

    def test_does_not_remove_acceptable_quality(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("ok prompt", quality=0.5)
        removed = mem.evict_bad_memories()
        assert removed == 0

    def test_boundary_quality_019_removed(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("borderline", quality=0.19)
        assert mem.evict_bad_memories() == 1

    def test_boundary_quality_02_kept(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("borderline", quality=0.20)
        assert mem.evict_bad_memories() == 0


# ---------------------------------------------------------------------------
# top_working — query boosting
# ---------------------------------------------------------------------------

class TestTopWorkingQuery:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_query_boosts_relevant_entry_above_high_quality_irrelevant(self, tmp_path):
        # "qqq zzz" has zero character-bigram overlap with "cherry blossom festival",
        # so has_match=False → the lower-quality cherry entry wins via (True, ·) > (False, ·).
        mem = self._mem(tmp_path)
        mem.add_entry("cherry blossom petal falls gently", quality=0.5)
        mem.add_entry("qqq zzz",                           quality=0.9)
        top = mem.top_working(1, query="cherry blossom festival")
        assert "cherry" in top[0].content

    def test_no_query_returns_by_priority_only(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("low quality entry",  quality=0.1)
        mem.add_entry("high quality entry", quality=1.0)
        top = mem.top_working(1)
        assert "high quality" in top[0].content

    def test_top_n_respected(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(10):
            mem.add_entry(f"entry {i}", quality=float(i) / 10)
        assert len(mem.top_working(3)) == 3

    def test_empty_working_returns_empty(self, tmp_path):
        mem = self._mem(tmp_path)
        assert mem.top_working(5, query="anything") == []


# ---------------------------------------------------------------------------
# promote_from_archive
# ---------------------------------------------------------------------------

class TestPromoteFromArchive:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_archive_cleared_after_promote(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.archive.append(MemoryEntry.from_prompt("archived entry"))
        mem.promote_from_archive([MemoryEntry.from_prompt("distilled insight")])
        assert len(mem.archive) == 0

    def test_promoted_entries_added_to_l2(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.archive.append(MemoryEntry.from_prompt("archived"))
        promoted = [MemoryEntry.from_prompt("insight A"), MemoryEntry.from_prompt("insight B")]
        mem.promote_from_archive(promoted)
        contents = [e.content for e in mem.working]
        assert "insight A" in contents
        assert "insight B" in contents

    def test_promote_evicts_l2_when_full(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(L2_MAX):
            mem.add_entry(f"existing {i}", quality=0.5)
        assert len(mem.working) == L2_MAX
        mem.promote_from_archive([MemoryEntry.from_prompt("new distilled")])
        assert len(mem.working) == L2_MAX  # stayed at cap
        assert "new distilled" in [e.content for e in mem.working]


# ---------------------------------------------------------------------------
# should_distill_archive
# ---------------------------------------------------------------------------

class TestShouldDistillArchive:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_false_below_threshold(self, tmp_path):
        from memory import L3_DISTILL_AT
        mem = self._mem(tmp_path)
        for i in range(L3_DISTILL_AT - 1):
            mem.archive.append(MemoryEntry.from_prompt(f"archived {i}"))
        assert mem.should_distill_archive() is False

    def test_true_at_threshold(self, tmp_path):
        from memory import L3_DISTILL_AT
        mem = self._mem(tmp_path)
        for i in range(L3_DISTILL_AT):
            mem.archive.append(MemoryEntry.from_prompt(f"archived {i}"))
        assert mem.should_distill_archive() is True


# ---------------------------------------------------------------------------
# should_distill_working — edge cases
# ---------------------------------------------------------------------------

class TestShouldDistillWorkingEdgeCases:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_false_when_not_multiple_of_5(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(6):
            mem.add_entry(f"good {i}", quality=0.8)
        assert mem.should_distill_working() is False  # 6 is not a multiple of 5

    def test_false_when_count_not_above_last_distill_count(self, tmp_path):
        mem = self._mem(tmp_path)
        for i in range(5):
            mem.add_entry(f"good {i}", quality=0.8)
        mem._last_distill_good_count = 5  # pretend we already distilled at 5
        assert mem.should_distill_working() is False

    def test_true_at_next_multiple_of_5_after_last(self, tmp_path):
        mem = self._mem(tmp_path)
        mem._last_distill_good_count = 5
        for i in range(10):
            mem.add_entry(f"good {i}", quality=0.8)
        assert mem.should_distill_working() is True  # 10 > 5, 10 % 5 == 0


# ---------------------------------------------------------------------------
# record_rating — fallback content-match when _last_entry is None
# ---------------------------------------------------------------------------

class TestRecordRatingFallback:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_fallback_to_content_match_when_last_entry_none(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("target prompt", quality=0.5)
        mem._last_entry = None  # simulate restarted process
        mem.record_rating("target prompt", 5)
        assert mem.working[0].quality == 1.0

    def test_no_crash_when_prompt_not_found(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("some other prompt", quality=0.5)
        mem._last_entry = None
        mem.record_rating("nonexistent prompt", 5)  # should not raise
        assert mem.working[0].quality == 0.5  # unchanged


# ---------------------------------------------------------------------------
# update() — allowed vs disallowed fields
# ---------------------------------------------------------------------------

class TestMemoryUpdate:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_allowed_fields_updated(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.update(language="en", notes="user prefers dark mood")
        assert mem.language == "en"
        assert mem.notes == "user prefers dark mood"

    def test_disallowed_fields_silently_ignored(self, tmp_path):
        mem = self._mem(tmp_path)
        original_count = mem.conversation_count
        mem.update(conversation_count=9999, working=[])  # not in allowed set
        assert mem.conversation_count == original_count
        assert mem.working == []  # list was already empty, not set to []


# ---------------------------------------------------------------------------
# Old-format migration
# ---------------------------------------------------------------------------

class TestOldFormatMigration:
    def test_successful_prompts_migrated_to_working(self, tmp_path):
        import json
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "successful_prompts": ["old prompt A", "old prompt B"],
        }), encoding="utf-8")
        mem = UserMemory(path).load()
        contents = [e.content for e in mem.working]
        assert "old prompt A" in contents
        assert "old prompt B" in contents

    def test_old_ratings_applied_to_working(self, tmp_path):
        import json
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "successful_prompts": ["rated prompt"],
            "ratings": [{"prompt": "rated prompt", "rating": 4}],
        }), encoding="utf-8")
        mem = UserMemory(path).load()
        assert mem.working[0].quality == 0.8  # rating 4 → 0.8

    def test_no_duplicate_migration(self, tmp_path):
        import json
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "successful_prompts": ["same", "same"],
        }), encoding="utf-8")
        mem = UserMemory(path).load()
        assert len([e for e in mem.working if e.content == "same"]) == 1


# ---------------------------------------------------------------------------
# as_context_str — query integration
# ---------------------------------------------------------------------------

class TestAsContextStr:
    def _mem(self, tmp_path):
        return UserMemory(tmp_path / "mem.json").load()

    def test_includes_language_and_resolution(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.update(language="en")
        ctx = mem.as_context_str()
        assert "Language: en" in ctx
        assert "1280" in ctx

    def test_includes_notes_when_set(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.update(notes="user dislikes desaturated palettes")
        assert "desaturated" in mem.as_context_str()

    def test_omits_notes_when_empty(self, tmp_path):
        mem = self._mem(tmp_path)
        assert "Notes" not in mem.as_context_str()

    def test_query_surfaces_relevant_memory_in_context(self, tmp_path):
        mem = self._mem(tmp_path)
        mem.add_entry("mountain snowfall at dawn",   quality=0.5)
        mem.add_entry("cherry blossom spring scene", quality=0.9)
        ctx = mem.as_context_str(query="snowy mountain")
        # the mountain entry should appear despite lower quality
        assert "mountain" in ctx
