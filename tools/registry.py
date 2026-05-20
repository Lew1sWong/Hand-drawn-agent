import os

from tools.base import BaseTool
from tools.video.image_to_video import ImageToVideoTool
from tools.video.text_to_video import TextToVideoTool
from tools.video.multi_shot import MultiShotTool
from tools.video.figurine import FigurineToAnimeCharTool
from tools.audio.tts import TTSTool
from tools.audio.audio_portrait import AudioPortraitTool
from tools.audio.add_bgm import AddBGMTool

_ALL_TOOLS_LIST: list[BaseTool] = [
    FigurineToAnimeCharTool(),
    ImageToVideoTool(),
    TextToVideoTool(),
    MultiShotTool(),
    TTSTool(),
    AudioPortraitTool(),
    AddBGMTool(),
]


def _load_tools() -> list[BaseTool]:
    """Return tools filtered by ENABLED_TOOLS env var (comma-separated names).
    If ENABLED_TOOLS is unset, all tools are enabled."""
    enabled_env = os.environ.get("ENABLED_TOOLS", "").strip()
    if not enabled_env:
        return _ALL_TOOLS_LIST
    names = {n.strip() for n in enabled_env.split(",") if n.strip()}
    return [t for t in _ALL_TOOLS_LIST if t.name in names]


ALL_TOOLS: list[BaseTool] = _load_tools()
TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}
