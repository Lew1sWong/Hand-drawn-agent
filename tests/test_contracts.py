"""
Verify every registered tool has a ToolContract and that the declared keys
are internally consistent (no typos, no missing required declarations).
No API calls required.
"""
import pytest
from tools import ALL_TOOLS, TOOL_MAP
from tools.base import ToolContract


class TestToolContracts:
    @pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
    def test_has_contract(self, tool):
        assert hasattr(tool, "contract"), f"{tool.name} is missing a ToolContract"
        assert isinstance(tool.contract, ToolContract)

    @pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
    def test_contract_reads_are_strings(self, tool):
        for key in tool.contract.reads:
            assert isinstance(key, str), f"{tool.name}.contract.reads contains non-string: {key!r}"

    @pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
    def test_contract_writes_are_strings(self, tool):
        for key in tool.contract.writes:
            assert isinstance(key, str), f"{tool.name}.contract.writes contains non-string: {key!r}"

    @pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
    def test_no_overlap_reads_writes(self, tool):
        # A key in both reads and writes is usually an overwrite (ok for add_bgm video_url,
        # figurine image_url) — but optional_reads should not duplicate required reads.
        required = set(tool.contract.reads)
        optional = set(tool.contract.optional_reads)
        overlap = required & optional
        assert not overlap, (
            f"{tool.name}: keys appear in both reads and optional_reads: {overlap}"
        )

    def test_tool_map_matches_all_tools(self):
        assert set(TOOL_MAP.keys()) == {t.name for t in ALL_TOOLS}

    def test_check_ctx_raises_on_missing_key(self):
        # pick any tool with required reads — image_to_video needs image_url + enhanced_prompt
        from tools.video.image_to_video import ImageToVideoTool
        from tools.base import ToolError
        tool = ImageToVideoTool()
        with pytest.raises(ToolError, match="Missing required context keys"):
            tool.check_ctx({})

    def test_check_ctx_passes_with_all_keys(self):
        from tools.video.image_to_video import ImageToVideoTool
        tool = ImageToVideoTool()
        tool.check_ctx({"image_url": "http://x.com/a.jpg", "enhanced_prompt": "test"})

    def test_add_bgm_contract(self):
        from tools.audio.add_bgm import AddBGMTool
        t = AddBGMTool()
        assert "video_url"        in t.contract.reads
        assert "user_description" in t.contract.reads
        assert "mode"             in t.contract.optional_reads
        assert "video_url"        in t.contract.writes
        assert "has_audio"        in t.contract.writes

    def test_audio_portrait_contract(self):
        from tools.audio.audio_portrait import AudioPortraitTool
        t = AudioPortraitTool()
        assert "image_url"  in t.contract.reads
        assert "audio_url"  in t.contract.reads
        assert "video_url"  in t.contract.writes
