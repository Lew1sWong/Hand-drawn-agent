"""
Unit tests for planner/rules.py — no API calls, no env vars required.
"""
import pytest
from planner.rules import AudioMode, VideoMode, classify


# ---------------------------------------------------------------------------
# Video mode routing
# ---------------------------------------------------------------------------

class TestVideoMode:
    def test_image_only(self):
        r = classify("给我做个动画", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.IMAGE_TO_VIDEO
        assert r.confidence == 1.0

    def test_no_image_no_keywords_defaults_to_text_to_video(self):
        r = classify("生成一个宇宙飞船的视频", has_image=False, has_audio=False)
        assert r.confidence == 1.0
        assert r.video_mode == VideoMode.TEXT_TO_VIDEO

    def test_image_plus_audio_always_portrait(self):
        r = classify("随便什么描述", has_image=True, has_audio=True)
        assert r.video_mode == VideoMode.AUDIO_PORTRAIT
        assert r.confidence == 1.0

    def test_multi_shot_keyword_zh(self):
        r = classify("帮我做个多镜头视频", has_image=False, has_audio=False)
        assert r.video_mode == VideoMode.MULTI_SHOT
        assert r.confidence == 1.0

    def test_multi_shot_keyword_en(self):
        r = classify("make a multi-shot animation", has_image=False, has_audio=False)
        assert r.video_mode == VideoMode.MULTI_SHOT
        assert r.confidence == 1.0

    def test_multi_shot_scene_count(self):
        r = classify("场景1: 开场  场景2: 战斗  场景3: 结局", has_image=False, has_audio=False)
        assert r.video_mode == VideoMode.MULTI_SHOT
        assert r.n_shots == 3

    def test_multi_shot_minimum_n_shots(self):
        # No explicit scene numbers → minimum is 2
        r = classify("做个分镜视频", has_image=False, has_audio=False)
        assert r.n_shots == 2

    def test_figurine_with_image(self):
        r = classify("把这个手办做成动画", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.FIGURINE
        assert r.confidence == 1.0

    def test_figurine_without_image_falls_through(self):
        # No image → can't do figurine → defaults to text_to_video
        r = classify("把手办做成动画", has_image=False, has_audio=False)
        assert r.video_mode == VideoMode.TEXT_TO_VIDEO
        assert r.confidence == 1.0

    def test_figurine_keyword_en(self):
        r = classify("animate this figurine", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.FIGURINE

    def test_audio_portrait_takes_priority_over_figurine(self):
        # image + audio → AUDIO_PORTRAIT wins, even if figurine keyword present
        r = classify("把这个手办配上音频", has_image=True, has_audio=True)
        assert r.video_mode == VideoMode.AUDIO_PORTRAIT


# ---------------------------------------------------------------------------
# Audio mode routing
# ---------------------------------------------------------------------------

class TestAudioMode:
    def test_sfx_keyword_zh(self):
        r = classify("加上背景音效", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.SFX
        assert r.wants_audio is True

    def test_sfx_keyword_engine(self):
        r = classify("摩托车引擎声", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.SFX

    def test_sfx_keyword_wind(self):
        r = classify("add wind sound", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.SFX

    def test_sfx_keyword_ambient(self):
        r = classify("ambient background music", has_image=False, has_audio=False)
        assert r.audio_mode == AudioMode.SFX
        assert r.wants_audio is True

    def test_narration_keyword_zh(self):
        r = classify("加上旁白", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.NARRATION
        assert r.wants_audio is True

    def test_narration_keyword_read_aloud(self):
        r = classify("朗读这段文字", has_image=False, has_audio=False)
        assert r.audio_mode == AudioMode.NARRATION

    def test_narration_keyword_voice_over(self):
        r = classify("add a voice over", has_image=False, has_audio=False)
        assert r.audio_mode == AudioMode.NARRATION

    def test_ambiguous_sound_wants_audio_true(self):
        # "声音" is ambiguous — not SFX, not NARRATION, but wants_audio=True
        r = classify("加上声音", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.UNKNOWN
        assert r.wants_audio is True

    def test_ambiguous_peiyin_wants_audio_true(self):
        r = classify("加配音", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.UNKNOWN
        assert r.wants_audio is True

    def test_no_audio_keywords(self):
        r = classify("做个动画", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.UNKNOWN
        assert r.wants_audio is False

    def test_sfx_and_narration_both_present_yields_mixed(self):
        # Both keyword families present → MIXED mode
        r = classify("加上音效旁白", has_image=True, has_audio=False)
        assert r.audio_mode == AudioMode.MIXED
        assert r.wants_audio is True

    def test_yinshi_classified_as_narration(self):
        # "吟诗" should trigger NARRATION mode
        r = classify('人物在仙鹤上吟诗："扶摇直上九万里！"', has_image=False, has_audio=False)
        assert r.audio_mode == AudioMode.NARRATION
        assert r.wants_audio is True

    def test_langsung_classified_as_narration(self):
        r = classify("让角色朗诵这首诗", has_image=False, has_audio=False)
        assert r.audio_mode == AudioMode.NARRATION

    def test_full_crane_prompt_yields_mixed(self):
        # The original failing prompt: has both 音效 (SFX) and 吟诗 (NARRATION)
        prompt = (
            '画面：让仙鹤动起来，有穿过云雾的动态效果，人物在仙鹤上吟诗："扶摇直上九万里！" '
            '音效：仙鹤扇动翅膀的声音，和人物吟诗："扶摇直上九万里！"'
        )
        r = classify(prompt, has_image=False, has_audio=False)
        assert r.video_mode == VideoMode.TEXT_TO_VIDEO
        assert r.audio_mode == AudioMode.MIXED
        assert r.wants_audio is True


# ---------------------------------------------------------------------------
# Combined scenarios
# ---------------------------------------------------------------------------

class TestCombined:
    def test_image_plus_sfx(self):
        r = classify("给图片加上背景音效", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.IMAGE_TO_VIDEO
        assert r.audio_mode == AudioMode.SFX
        assert r.confidence == 1.0

    def test_image_plus_narration(self):
        r = classify("帮图片加上旁白", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.IMAGE_TO_VIDEO
        assert r.audio_mode == AudioMode.NARRATION

    def test_figurine_plus_sfx(self):
        r = classify("把手办做成动画并加上环境音", has_image=True, has_audio=False)
        assert r.video_mode == VideoMode.FIGURINE
        assert r.audio_mode == AudioMode.SFX
        assert r.wants_audio is True

    def test_audio_portrait_no_bgm(self):
        # audio_portrait: wants_audio is irrelevant but audio_mode may still be set
        r = classify("加上旁白配音", has_image=True, has_audio=True)
        assert r.video_mode == VideoMode.AUDIO_PORTRAIT
        # audio_mode detected from text but video mode is portrait (no add_bgm needed)
