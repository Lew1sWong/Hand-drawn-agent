"""
Unit tests for the conversation gateway.
No real API calls — DeepSeek is mocked throughout.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conversation import (
    ConversationSession, ConvDecision, PostGenDecision, _fallback_reply,
)


# ---------------------------------------------------------------------------
# ConversationSession — history management
# ---------------------------------------------------------------------------

class TestConversationSessionHistory:
    def test_initial_state(self):
        s = ConversationSession("u1")
        assert s.exchange_count == 0
        assert s._history == []

    def test_add_user_increments_exchange_count(self):
        s = ConversationSession("u1")
        s.add_user("hello")
        assert s.exchange_count == 1

    def test_add_assistant_does_not_increment_exchange_count(self):
        s = ConversationSession("u1")
        s.add_user("hello")
        s.add_assistant("hi")
        assert s.exchange_count == 1

    def test_multiple_turns(self):
        s = ConversationSession("u1")
        s.add_user("msg1");  s.add_assistant("r1")
        s.add_user("msg2");  s.add_assistant("r2")
        assert s.exchange_count == 2

    def test_clear_resets_history(self):
        s = ConversationSession("u1")
        s.add_user("hello");  s.add_assistant("hi")
        s.clear()
        assert s._history == []
        assert s.exchange_count == 0

    def test_trim_keeps_max_history(self):
        s = ConversationSession("u1")
        for i in range(20):
            s._history.append({"role": "user", "content": f"msg {i}"})
        s._trim()
        assert len(s._history) == s._MAX_HISTORY

    def test_trim_keeps_most_recent(self):
        s = ConversationSession("u1")
        for i in range(20):
            s._history.append({"role": "user", "content": f"msg {i}"})
        s._trim()
        assert s._history[-1]["content"] == "msg 19"

    def test_trim_triggered_automatically_on_add(self):
        s = ConversationSession("u1")
        for i in range(s._MAX_HISTORY + 5):
            s.add_user(f"msg {i}")
        assert len(s._history) <= s._MAX_HISTORY


# ---------------------------------------------------------------------------
# ConversationSession.process — LLM integration
# ---------------------------------------------------------------------------

def _make_mock_client(action: str, reply: str, user_request: str = ""):
    """Helper: build a mock AsyncOpenAI client that returns the given decision."""
    payload = {"action": action, "reply": reply, "user_request": user_request}
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=mock_resp)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client


class TestConversationSessionProcess:
    @pytest.mark.asyncio
    async def test_execute_action_returned(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client(
            "execute", "好的，马上生成！", "夕阳下樱花飘落的手绘动画"
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            decision = await s.process("做个樱花场景", {})

        assert decision.action == "execute"
        assert decision.reply == "好的，马上生成！"
        assert "樱花" in decision.user_request

    @pytest.mark.asyncio
    async def test_chat_action_returned(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client("chat", "请描述一下想要的场景？", "")
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            decision = await s.process("做个视频", {})

        assert decision.action == "chat"
        assert "场景" in decision.reply
        assert decision.user_request == "做个视频"   # raw fallback when LLM returns empty

    @pytest.mark.asyncio
    async def test_history_grows_per_process_call(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client("chat", "请描述场景？")
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            await s.process("做个视频", {})

        # 1 user message + 1 assistant message
        assert len(s._history) == 2
        assert s._history[0]["role"] == "user"
        assert s._history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_process_includes_history_in_llm_call(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client("execute", "好的！", "古风场景")

        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            # First turn
            await s.process("我想要古风", {})
            # Second turn
            await s.process("小桥流水人家", {})

        # The second LLM call should have received 3 messages (2 from turn 1 + 1 from turn 2)
        calls = mock_client.chat.completions.create.call_args_list
        last_call_messages = calls[-1].kwargs["messages"]
        # system prompt is index 0; history follows
        history_messages = [m for m in last_call_messages if m["role"] != "system"]
        assert len(history_messages) >= 3

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_execute(self):
        s = ConversationSession("u1")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            decision = await s.process("做个夕阳海边动画", {})

        assert decision.action == "execute"
        assert decision.user_request == "做个夕阳海边动画"  # raw message used as fallback

    @pytest.mark.asyncio
    async def test_json_parse_failure_falls_back(self):
        s = ConversationSession("u1")
        mock_msg = MagicMock()
        mock_msg.content = "not valid json {{{"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            decision = await s.process("做个视频", {})

        assert decision.action == "execute"

    @pytest.mark.asyncio
    async def test_available_assets_passed_in_system_prompt(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client("execute", "好的", "animate this image")
        captured_system = []

        async def capture_create(**kwargs):
            captured_system.append(kwargs["messages"][0]["content"])
            payload = {"action": "execute", "reply": "好的", "user_request": "animate this image"}
            mock_msg = MagicMock()
            mock_msg.content = json.dumps(payload)
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_resp = MagicMock()
            mock_resp.choices = [mock_choice]
            return mock_resp

        mock_client.chat.completions.create = capture_create
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            await s.process("帮我动画化", {"image_url": "http://x.com/a.jpg"})

        assert "image_url" in captured_system[0]

    @pytest.mark.asyncio
    async def test_user_request_falls_back_to_raw_message_when_empty(self):
        s = ConversationSession("u1")
        mock_client = _make_mock_client("execute", "好的！", "")  # empty user_request
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            decision = await s.process("做个场景", {})

        assert decision.user_request == "做个场景"


# ---------------------------------------------------------------------------
# _fallback_reply helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PostGenDecision — process_post_gen
# ---------------------------------------------------------------------------

def _make_post_gen_mock(action: str, reply: str, **extra):
    payload = {"action": action, "reply": reply, **extra}
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(payload)
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(return_value=mock_resp)
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client


class TestProcessPostGen:
    @pytest.mark.asyncio
    async def test_regenerate_action(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "regenerate",
            "好的，正在重新生成，画面会更暗一点……",
            refined_request="夕阳下奔跑的少女，画面偏暗，剪影效果",
            param_overrides={},
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("能不能更暗一点", "夕阳下奔跑的少女", {})

        assert d.action == "regenerate"
        assert "暗" in d.refined_request
        assert d.param_overrides == {}

    @pytest.mark.asyncio
    async def test_duration_override_extracted(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "regenerate",
            "好的，改成8秒！",
            refined_request="夕阳下奔跑的少女",
            param_overrides={"duration": 8},
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("时长改成8秒", "夕阳下奔跑的少女", {})

        assert d.action == "regenerate"
        assert d.param_overrides == {"duration": 8}

    @pytest.mark.asyncio
    async def test_duration_clamped_to_valid_values(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "regenerate", "好的",
            refined_request="test",
            param_overrides={"duration": 6},   # 6 is not 4 or 8 → clamp to 8
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("改长一点", "test prompt", {})

        assert d.param_overrides["duration"] == 8   # rounded up to 8

    @pytest.mark.asyncio
    async def test_add_bgm_sfx(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "add_bgm", "好的，添加环境音效！",
            bgm_mode="sfx",
            bgm_description="风声和海浪声",
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("加点风声和海浪声", "海边动画", {"image_url": "x"})

        assert d.action == "add_bgm"
        assert d.bgm_mode == "sfx"
        assert "海浪" in d.bgm_description

    @pytest.mark.asyncio
    async def test_add_bgm_narration(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "add_bgm", "好的，加旁白！",
            bgm_mode="narration",
            bgm_description="一段诗意的旁白配音",
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("加个旁白", "海边动画", {})

        assert d.action == "add_bgm"
        assert d.bgm_mode == "narration"

    @pytest.mark.asyncio
    async def test_rate_numeric_extracted(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "rate", "谢谢！",
            rating=4,
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("不错，4分", "some prompt", {})

        assert d.action == "rate"
        assert d.rating == 4

    @pytest.mark.asyncio
    async def test_rate_semantic_extracted(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "rate", "谢谢！",
            rating=5,
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("非常满意！", "some prompt", {})

        assert d.action == "rate"
        assert d.rating == 5

    @pytest.mark.asyncio
    async def test_rating_clamped_to_1_5(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock("rate", "ok", rating=9)   # out-of-range
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("给10分", "some prompt", {})

        assert d.rating == 5   # clamped to max

    @pytest.mark.asyncio
    async def test_new_request_action(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "new_request", "好的，开始新的创作！",
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("换个完全不同的主题，做个龙的视频", "海边动画", {})

        assert d.action == "new_request"

    @pytest.mark.asyncio
    async def test_chat_action(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "chat", "这个AI是基于火山引擎的即梦模型生成的。",
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("这用了什么技术？", "some prompt", {})

        assert d.action == "chat"
        assert d.reply

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_chat(self):
        s = ConversationSession("u1")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("再暗一点", "some prompt", {})

        assert d.action == "chat"   # safe fallback — does NOT execute or regenerate

    @pytest.mark.asyncio
    async def test_last_prompt_injected_in_system(self):
        s = ConversationSession("u1")
        captured = []

        async def capture(**kwargs):
            captured.append(kwargs["messages"][0]["content"])
            payload = {"action": "chat", "reply": "ok"}
            msg = MagicMock(); msg.content = json.dumps(payload)
            c = MagicMock(); c.message = msg
            r = MagicMock(); r.choices = [c]
            return r

        mock_client = MagicMock()
        mock_client.chat.completions.create = capture
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            await s.process_post_gen("能更亮吗", "夕阳少女奔跑动画", {})

        assert "夕阳少女奔跑动画" in captured[0]

    @pytest.mark.asyncio
    async def test_history_grows_through_post_gen_turns(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock("chat", "好的")
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            await s.process_post_gen("改一下", "prompt A", {})
            await s.process_post_gen("再改一下", "prompt A", {})

        assert s.exchange_count == 2

    @pytest.mark.asyncio
    async def test_unsafe_param_overrides_stripped(self):
        s = ConversationSession("u1")
        mock_client = _make_post_gen_mock(
            "regenerate", "ok",
            refined_request="test",
            param_overrides={"duration": 8, "malicious_key": "bad_value"},
        )
        with patch("conversation.AsyncOpenAI", return_value=mock_client):
            d = await s.process_post_gen("改长", "test", {})

        assert "malicious_key" not in d.param_overrides
        assert d.param_overrides.get("duration") == 8


# ---------------------------------------------------------------------------
# _fallback_reply helper
# ---------------------------------------------------------------------------

class TestFallbackReply:
    def test_chinese_text_returns_chinese(self):
        assert "生成" in _fallback_reply("做个视频")

    def test_english_text_returns_english(self):
        reply = _fallback_reply("make a video")
        assert "generating" in reply.lower() or "Got it" in reply

    def test_mixed_text_with_chinese_returns_chinese(self):
        assert "生成" in _fallback_reply("sunset 场景")
