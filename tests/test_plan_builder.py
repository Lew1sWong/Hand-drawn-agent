"""
Unit tests for planner — plan builder, patcher, validator.
No API calls, no env vars required.
"""
import pytest
from planner.rules import AudioMode, VideoMode, RuleResult, classify
from planner.planner import _build_plan_from_rule, _patch_plan, _validate_plan, PlanStep


# ---------------------------------------------------------------------------
# _build_plan_from_rule
# ---------------------------------------------------------------------------

class TestBuildPlanFromRule:
    def _rule(self, video_mode, audio_mode=AudioMode.UNKNOWN, wants_audio=False, n_shots=None):
        return RuleResult(
            video_mode=video_mode,
            audio_mode=audio_mode,
            wants_audio=wants_audio,
            n_shots=n_shots,
            confidence=1.0,
        )

    def test_image_to_video_no_audio(self):
        plan = _build_plan_from_rule(self._rule(VideoMode.IMAGE_TO_VIDEO))
        assert len(plan) == 1
        assert plan[0].tool == "image_to_video"

    def test_image_to_video_plus_sfx(self):
        plan = _build_plan_from_rule(self._rule(
            VideoMode.IMAGE_TO_VIDEO, AudioMode.SFX, wants_audio=True
        ))
        assert [s.tool for s in plan] == ["image_to_video", "add_bgm"]
        assert plan[1].inputs["mode"] == "sfx"

    def test_image_to_video_plus_narration(self):
        plan = _build_plan_from_rule(self._rule(
            VideoMode.IMAGE_TO_VIDEO, AudioMode.NARRATION, wants_audio=True
        ))
        assert plan[1].inputs["mode"] == "narration"

    def test_image_to_video_ambiguous_audio_defaults_sfx(self):
        # wants_audio=True but mode=UNKNOWN → safe default is sfx
        plan = _build_plan_from_rule(self._rule(
            VideoMode.IMAGE_TO_VIDEO, AudioMode.UNKNOWN, wants_audio=True
        ))
        assert plan[1].tool == "add_bgm"
        assert plan[1].inputs["mode"] == "sfx"

    def test_figurine_no_audio(self):
        plan = _build_plan_from_rule(self._rule(VideoMode.FIGURINE))
        assert [s.tool for s in plan] == ["figurine_to_anime", "image_to_video"]

    def test_figurine_plus_sfx(self):
        plan = _build_plan_from_rule(self._rule(
            VideoMode.FIGURINE, AudioMode.SFX, wants_audio=True
        ))
        assert [s.tool for s in plan] == ["figurine_to_anime", "image_to_video", "add_bgm"]
        assert plan[2].inputs["mode"] == "sfx"

    def test_multi_shot_with_n_shots(self):
        plan = _build_plan_from_rule(self._rule(VideoMode.MULTI_SHOT, n_shots=3))
        assert len(plan) == 1
        assert plan[0].tool == "multi_shot_video"
        assert plan[0].inputs["n_shots"] == 3

    def test_multi_shot_no_n_shots_no_inputs(self):
        plan = _build_plan_from_rule(self._rule(VideoMode.MULTI_SHOT))
        assert plan[0].inputs == {}

    def test_audio_portrait_no_add_bgm(self):
        # Even if wants_audio=True, audio_portrait already handles audio
        plan = _build_plan_from_rule(self._rule(
            VideoMode.AUDIO_PORTRAIT, AudioMode.SFX, wants_audio=True
        ))
        assert len(plan) == 1
        assert plan[0].tool == "audio_portrait"

    def test_no_wants_audio_no_add_bgm(self):
        plan = _build_plan_from_rule(self._rule(
            VideoMode.IMAGE_TO_VIDEO, AudioMode.UNKNOWN, wants_audio=False
        ))
        assert all(s.tool != "add_bgm" for s in plan)


# ---------------------------------------------------------------------------
# _patch_plan — inject audio_mode into LLM-produced plans
# ---------------------------------------------------------------------------

class TestPatchPlan:
    def test_patches_sfx_mode(self):
        plan = [
            PlanStep(tool="text_to_video"),
            PlanStep(tool="add_bgm", inputs={}),
        ]
        rule = RuleResult(audio_mode=AudioMode.SFX)
        patched = _patch_plan(plan, rule)
        assert patched[1].inputs["mode"] == "sfx"

    def test_patches_narration_mode(self):
        plan = [PlanStep(tool="add_bgm", inputs={})]
        rule = RuleResult(audio_mode=AudioMode.NARRATION)
        patched = _patch_plan(plan, rule)
        assert patched[0].inputs["mode"] == "narration"

    def test_no_patch_when_audio_unknown(self):
        plan = [PlanStep(tool="add_bgm", inputs={"mode": "sfx"})]
        rule = RuleResult(audio_mode=AudioMode.UNKNOWN)
        patched = _patch_plan(plan, rule)
        assert patched[0].inputs["mode"] == "sfx"  # original preserved

    def test_only_patches_add_bgm(self):
        plan = [
            PlanStep(tool="image_to_video", inputs={}),
            PlanStep(tool="add_bgm", inputs={}),
        ]
        rule = RuleResult(audio_mode=AudioMode.SFX)
        patched = _patch_plan(plan, rule)
        assert patched[0].inputs == {}  # image_to_video untouched
        assert patched[1].inputs["mode"] == "sfx"

    def test_no_add_bgm_in_plan_is_noop(self):
        plan = [PlanStep(tool="text_to_video")]
        rule = RuleResult(audio_mode=AudioMode.SFX)
        patched = _patch_plan(plan, rule)
        assert len(patched) == 1


# ---------------------------------------------------------------------------
# _validate_plan
# ---------------------------------------------------------------------------

class TestValidatePlan:
    def test_valid_plan_passes_through(self):
        plan = [PlanStep(tool="image_to_video"), PlanStep(tool="add_bgm")]
        validated = _validate_plan(plan)
        assert len(validated) == 2

    def test_strips_hallucinated_tools(self):
        plan = [
            PlanStep(tool="image_to_video"),
            PlanStep(tool="nonexistent_tool_xyz"),
        ]
        validated = _validate_plan(plan)
        assert len(validated) == 1
        assert validated[0].tool == "image_to_video"

    def test_all_hallucinated_raises(self):
        plan = [PlanStep(tool="fake_tool"), PlanStep(tool="another_fake")]
        with pytest.raises(ValueError, match="no valid steps"):
            _validate_plan(plan)

    def test_empty_plan_raises(self):
        with pytest.raises(ValueError):
            _validate_plan([])
