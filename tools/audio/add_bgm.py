from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

import edge_tts
from openai import AsyncOpenAI

from tools.base import BaseTool, ToolContract
from tools._replicate import PUBLIC_BASE_URL, _replicate_to_url

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio planning helper
# ---------------------------------------------------------------------------

_AUDIO_DIRECTOR_SYSTEM = """\
You are an audio director for short animated videos (4–8 seconds).
{mode_instruction}

Always return this JSON shape (all fields required):
{{
  "mode": "ambient" | "narration",
  "ambient_prompt": "<English sound-design prompt, 20–50 words, sounds only — required for ambient, empty string for narration>",
  "duration_s": <integer 8–15, required for ambient, 0 for narration>,
  "narration_text": "<1–3 sentence poetic scene narration in the user's language — ALWAYS fill this, it is used as TTS fallback if sfx generation fails>",
  "voice": "<edge-tts voice name>"
}}

Rules:
- ambient_prompt MUST be in English, describe SOUNDS ONLY (not visuals)
- Extract audio cues from the user's description; ignore camera moves, character appearance, visual details
- Translate audio intent into a stable-audio-open compatible English prompt (concrete sounds, not adjectives about mood)
- narration_text is a POETIC SCENE NARRATION written by you — do NOT copy the user's raw
  description. Imagine you are the narrator of a short film and write 1–3 sentences that
  evoke the mood and atmosphere of the scene. This field is ALWAYS required.
- Narration voices: zh-CN-XiaoxiaoNeural (zh ♀), zh-CN-YunxiNeural (zh ♂),
  en-US-JennyNeural (en ♀), en-US-GuyNeural (en ♂)
- Output ONLY the JSON object — no markdown, no explanation."""

_MODE_FREE = """\
Analyse the user's scene description and decide the best audio approach.

Mode decision:
- "ambient"   if user mentions: engine/motor sounds, animal sounds, nature sounds,
  wind, background noise, atmosphere, music, sound effects
  (音效 / 引擎声 / 背景声 / 环境音 / 风声 / 狗叫 / 鸟鸣)
- "narration" if user mentions: voice-over, character speech, narration, reading text aloud
  (配音 / 旁白 / 说话 / 朗读)
- DEFAULT to "ambient" when intent is ambiguous — never default to "narration"."""

_MODE_LOCKED_SFX = """\
Mode is LOCKED to "ambient" by the caller — do NOT change it.
Your job: extract the audio description from the user's message and translate/expand it
into a high-quality English sound-design prompt for stable-audio-open.
Focus only on sounds (engine noise, animal sounds, wind, etc.); ignore visual/camera directions."""

_MODE_LOCKED_NARRATION = """\
Mode is LOCKED to "narration" by the caller — do NOT change it.
Your job: generate a 1–3 sentence poetic narration of the scene in the user's language."""


async def _plan_audio(user_description: str, lang: str, hint_mode: str | None = None) -> dict:
    """
    Decide audio mode (ambient sfx vs narration) and generate the audio plan.

    hint_mode is set by the planner's rule layer when intent is unambiguous.
    When set, the mode decision is skipped but the LLM still generates a quality
    ambient_prompt / narration_text from the full scene description.

    Returns a dict: mode, ambient_prompt, duration_s, narration_text, voice.
    """
    if hint_mode == "sfx":
        mode_instruction = _MODE_LOCKED_SFX
    elif hint_mode == "narration":
        mode_instruction = _MODE_LOCKED_NARRATION
    else:
        mode_instruction = _MODE_FREE

    system = _AUDIO_DIRECTOR_SYSTEM.format(mode_instruction=mode_instruction)

    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"user_lang={lang}\n\n{user_description}"},
        ],
    )
    import json as _json
    raw = resp.choices[0].message.content.strip()
    logger.info("[AddBGM] audio plan: %s", raw)
    return _json.loads(raw)


async def _freesound_search_download(query: str, api_key: str) -> bytes | None:
    """Search Freesound for a sound, return preview MP3 bytes or None."""
    import requests as _req
    loop = asyncio.get_running_loop()
    try:
        def _search():
            return _req.get(
                "https://freesound.org/apiv2/search/text/",
                params={
                    "query":     query,
                    "token":     api_key,
                    "fields":    "id,name,previews,duration",
                    "filter":    "duration:[2 TO 60]",
                    "sort":      "rating_desc",
                    "page_size": 1,
                },
                timeout=10,
            )
        resp = await loop.run_in_executor(None, _search)
        results = resp.json().get("results", [])
        if not results:
            logger.warning("[AddBGM] Freesound: no results for %r", query)
            return None
        preview_url = results[0]["previews"]["preview-hq-mp3"]
        name = results[0]["name"]
        logger.info("[AddBGM] Freesound matched %r → %s", query, name)
        r = await loop.run_in_executor(None, lambda: _req.get(preview_url, timeout=30))
        r.raise_for_status()
        return r.content
    except Exception as exc:
        logger.warning("[AddBGM] Freesound search failed for %r: %s", query, exc)
        return None


async def _extract_sound_queries(ambient_prompt: str) -> list[str]:
    """Use DeepSeek to extract 2-3 Freesound-searchable keywords from ambient prompt."""
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    import json as _json
    resp = await client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=80,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "Extract 2-3 short English search queries for Freesound.org from this sound description. "
                "Each query should be 2-4 words that would find a matching sound recording. "
                'Return JSON: {"queries": ["query1", "query2", ...]}'
            )},
            {"role": "user", "content": ambient_prompt},
        ],
    )
    data = _json.loads(resp.choices[0].message.content)
    queries = data.get("queries", [])
    logger.info("[AddBGM] Freesound queries: %s", queries)
    return queries or [ambient_prompt[:50]]


async def _generate_ambient_audio(prompt: str, duration_s: int) -> bytes:
    """
    Generate ambient/sfx audio via Freesound (free real recordings) + ffmpeg mix.
    Falls back to Replicate stable-audio-open if FREESOUND_API_KEY not set.
    """
    import subprocess
    import requests as _req

    freesound_key = os.environ.get("FREESOUND_API_KEY", "")

    if freesound_key:
        # ── Freesound path (free, real recordings) ────────────────────────
        queries = await _extract_sound_queries(prompt)
        sound_files: list[Path] = []
        for query in queries[:3]:
            data = await _freesound_search_download(query, freesound_key)
            if data:
                p = Path(f"/tmp/sfx_{uuid.uuid4().hex}.mp3")
                p.write_bytes(data)
                sound_files.append(p)

        if not sound_files:
            raise RuntimeError("Freesound returned no results for any query")

        # Mix all found sounds and trim to duration
        uid = uuid.uuid4().hex
        out = Path(f"/tmp/sfx_mixed_{uid}.mp3")
        try:
            if len(sound_files) == 1:
                inputs  = ["-i", str(sound_files[0])]
                filters = f"atrim=duration={duration_s},apad=whole_dur={duration_s}"
                cmd = ["ffmpeg", "-y"] + inputs + ["-af", filters, str(out)]
            else:
                inputs  = []
                for f in sound_files:
                    inputs += ["-i", str(f)]
                n = len(sound_files)
                filters = f"amix=inputs={n}:duration=longest,atrim=duration={duration_s},apad=whole_dur={duration_s}"
                cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filters, str(out)]

            loop = asyncio.get_running_loop()
            proc = await loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, timeout=60)
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg mix error: {proc.stderr[-300:]}")

            logger.info("[AddBGM] Freesound mix done  size=%d", out.stat().st_size)
            return out.read_bytes()
        finally:
            for f in sound_files:
                f.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    else:
        # ── Replicate fallback (requires credits) ─────────────────────────
        import replicate
        api_token = os.environ.get("REPLICATE_API_TOKEN", "")
        if api_token:
            os.environ["REPLICATE_API_TOKEN"] = api_token

        loop = asyncio.get_running_loop()
        logger.info("[AddBGM] no FREESOUND_API_KEY — trying Replicate  prompt=%s", prompt[:80])
        output = await loop.run_in_executor(
            None,
            lambda: replicate.run(
                "meta/musicgen",
                input={
                    "prompt":                 f"ambient sound: {prompt}",
                    "duration":               min(int(duration_s), 30),
                    "model_version":          "stereo-large",
                    "normalization_strategy": "peak",
                    "output_format":          "mp3",
                },
            ),
        )
        audio_url = _replicate_to_url(output)
        r = await loop.run_in_executor(None, lambda: _req.get(audio_url, timeout=120))
        r.raise_for_status()
        return r.content


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class AddBGMTool(BaseTool):
    """
    Post-processing step: adds audio to a silent generated video.

    Pipeline:
      0. If planner set ctx["mode"], bypass LLM entirely; else call DeepSeek
      1. Download video from ctx["video_url"]
      2a. sfx/ambient mode  → Replicate stable-audio-open generates WAV
      2b. narration mode    → edge-tts synthesises voice-over
      3. Loop audio to match video length and merge with ffmpeg
      4. Serve merged file via /media/ and overwrite video_url

    Requires ffmpeg (brew install ffmpeg / apt install ffmpeg).
    """

    name        = "add_bgm"
    description = """\
Add audio to a silent video (image_to_video / text_to_video / multi_shot_video).

Uses DeepSeek to understand WHAT kind of audio the user wants:
- Ambient / sound effects (engine, wind, animals, music) → Replicate stable-audio-open
- Voice-over / narration / dialogue → edge-tts

Overwrites video_url with the audio+video file.

Add this as the LAST step when the user asks for sound / 声音 / 配音 / 音效 / 旁白 /
narration / audio / music / sound effects.
Do NOT add this after audio_portrait — that tool already produces audio.

Reads:  video_url, user_description, lang (optional, default zh)
Writes: video_url (the merged file served via /media/)"""

    input_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["sfx", "narration"],
                "description": (
                    "Audio mode override. "
                    "'sfx' = ambient/background/environmental sound via stable-audio-open. "
                    "'narration' = voice-over/speech via edge-tts. "
                    "DEFAULT to 'sfx' when ambiguous."
                ),
            },
            "voice": {
                "type": "string",
                "description": (
                    "edge-tts voice override (narration mode only). "
                    "Chinese: zh-CN-XiaoxiaoNeural (F), zh-CN-YunxiNeural (M). "
                    "English: en-US-JennyNeural (F), en-US-GuyNeural (M)."
                ),
            },
        },
    }
    contract = ToolContract(
        reads          = ["video_url", "user_description"],
        writes         = ["video_url", "has_audio"],
        optional_reads = ["mode", "lang", "voice", "tts_text"],
    )

    async def run(self, ctx: dict) -> dict:
        import subprocess
        import requests as _req

        video_url = ctx["video_url"]
        text      = ctx["user_description"]
        lang      = ctx.get("lang", "zh")
        default_voice = ctx.get("voice") or (
            "zh-CN-XiaoxiaoNeural" if lang == "zh" else "en-US-JennyNeural"
        )

        # ── Step 0: audio planning ─────────────────────────────────────────
        forced_mode = ctx.get("mode")  # "sfx" or "narration" injected by planner

        try:
            audio_plan = await _plan_audio(text, lang, hint_mode=forced_mode)
        except Exception as exc:
            logger.warning("[AddBGM] LLM planning failed (%s) — falling back to SFX", exc)
            audio_plan = {
                "mode":           "narration" if forced_mode == "narration" else "ambient",
                "ambient_prompt": "atmospheric background ambient sound",
                "duration_s":     12,
                "narration_text": "",
                "voice":          default_voice,
            }

        # Planner's mode decision is authoritative — override LLM if it disobeyed
        if forced_mode == "sfx":
            audio_plan["mode"] = "ambient"
        elif forced_mode == "narration":
            audio_plan["mode"] = "narration"

        mode = audio_plan.get("mode", "ambient")
        uid  = uuid.uuid4().hex

        audio_suffix = ".wav" if mode == "ambient" else ".mp3"
        audio_in  = Path(f"/tmp/bgm_audio_{uid}{audio_suffix}")
        video_in  = Path(f"/tmp/bgm_vid_in_{uid}.mp4")
        video_out = Path(f"/tmp/bgm_{uid}.mp4")

        try:
            # ── Step 1: download silent video ─────────────────────────────
            logger.info("[AddBGM] downloading video  url=%s", video_url)
            loop = asyncio.get_running_loop()
            last_exc: Exception | None = None
            for _attempt in range(3):
                try:
                    def _dl(u=video_url):
                        resp = _req.get(u, timeout=120)
                        resp.raise_for_status()
                        return resp.content
                    video_data = await loop.run_in_executor(None, _dl)
                    break
                except Exception as _e:
                    last_exc = _e
                    logger.warning("[AddBGM] video download attempt %d failed: %s", _attempt + 1, _e)
                    if _attempt < 2:
                        await asyncio.sleep(2 ** _attempt)
            else:
                raise RuntimeError(f"[AddBGM] video download failed after 3 attempts: {last_exc}")
            video_in.write_bytes(video_data)
            logger.info("[AddBGM] video saved  size=%d bytes", len(video_data))

            # ── Step 2: generate audio ─────────────────────────────────────
            if mode == "ambient":
                sfx_prompt = audio_plan.get("ambient_prompt", "ambient atmospheric sound")
                duration   = int(audio_plan.get("duration_s", 12))
                logger.info("[AddBGM] sfx prompt: %s", sfx_prompt)
                try:
                    audio_bytes = await _generate_ambient_audio(sfx_prompt, duration)
                    audio_in.write_bytes(audio_bytes)
                    logger.info("[AddBGM] sfx saved  size=%d bytes", len(audio_bytes))
                except Exception as exc:
                    # Stable-audio failed — fall back to TTS with LLM-written narration ONLY
                    # (never read the raw user prompt aloud)
                    logger.warning("[AddBGM] sfx generation failed (%s) — TTS fallback", exc)
                    narration_text = audio_plan.get("narration_text", "").strip()
                    if not narration_text:
                        # Last resort: short generic description, never the raw prompt
                        narration_text = "背景音效生成失败，请重试。" if lang == "zh" else "Background sound generation failed."
                    fallback_voice = audio_plan.get("voice", default_voice)
                    audio_in = Path(f"/tmp/bgm_audio_{uid}.mp3")
                    communicate = edge_tts.Communicate(narration_text, fallback_voice)
                    await communicate.save(str(audio_in))
            else:
                # Prefer verbatim quoted text extracted from user prompt (set by executor).
                # Fall back to LLM-generated narration only when no exact text is available.
                narration = (ctx.get("tts_text") or audio_plan.get("narration_text", "")).strip()
                if not narration:
                    narration = "背景音效生成失败，请重试。" if lang == "zh" else "Sound generation failed, please retry."
                voice = audio_plan.get("voice", default_voice)
                logger.info("[AddBGM] TTS  voice=%s  text=%s", voice, narration[:60])
                communicate = edge_tts.Communicate(narration, voice)
                await communicate.save(str(audio_in))
                logger.info("[AddBGM] TTS saved  size=%d bytes", audio_in.stat().st_size)

            # ── Step 3: ffmpeg merge ───────────────────────────────────────
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_in),
                "-stream_loop", "-1", "-i", str(audio_in),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                str(video_out),
            ]
            logger.info("[AddBGM] running ffmpeg  mode=%s", mode)
            _loop = asyncio.get_running_loop()
            proc = await _loop.run_in_executor(
                None, lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            )
            if proc.returncode != 0:
                raise RuntimeError(f"[AddBGM] ffmpeg error:\n{proc.stderr[-800:]}")
            logger.info("[AddBGM] merge done  size=%d bytes", video_out.stat().st_size)

            # ── Step 4: serve merged file ──────────────────────────────────
            out_name = f"bgm_{uid}.mp4"
            video_out.rename(Path(f"/tmp/{out_name}"))
            new_url = f"{PUBLIC_BASE_URL}/media/{out_name}"
            logger.info("[AddBGM] ready  url=%s", new_url)
            return {"video_url": new_url, "has_audio": True}

        finally:
            for p in (audio_in, video_in):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
