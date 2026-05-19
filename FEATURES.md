# 手绘动画 AI Agent — 功能全览
# Hand-Drawn Animation AI Agent — Complete Feature Guide

---

## 目录 / Contents

1. [唤醒与语言选择 / Wake & Language](#1-唤醒与语言选择--wake--language)
2. [视频生成模式 / Video Generation Modes](#2-视频生成模式--video-generation-modes)
3. [手办转动漫 / Figurine → Anime](#3-手办转动漫--figurine--anime)
4. [多镜头生成 / Multi-Shot Generation](#4-多镜头生成--multi-shot-generation)
5. [口播数字人 / Talking-Head Portrait](#5-口播数字人--talking-head-portrait)
6. [视频配音 / Add Narration to Video](#6-视频配音--add-narration-to-video)
7. [文字转语音 / Text-to-Speech](#7-文字转语音--text-to-speech)
8. [电影级运镜 / Cinematic Camera Language](#8-电影级运镜--cinematic-camera-language)
9. [后期处理框架 / Post-Production Commands](#9-后期处理框架--post-production-commands)
10. [取消与重置 / Cancel & Reset](#10-取消与重置--cancel--reset)
11. [评分反馈 / Rating & Feedback](#11-评分反馈--rating--feedback)
12. [三层记忆系统 / Three-Tier Memory](#12-三层记忆系统--three-tier-memory)
13. [架构概览 / Architecture Overview](#13-架构概览--architecture-overview)

---

## 1. 唤醒与语言选择 / Wake & Language

### 中文

发送任意唤醒词即可启动 Agent，并弹出交互式语言选择卡片：

> `hello` · `hi` · `你好` · `嗨` · `开始` · `start` · `help` · `帮助`

卡片提供两个按钮：**🇨🇳 中文** / **🇬🇧 English**，点击后所有后续回复切换为对应语言。

发送 `exit` · `退出` · `重置` · `reset` 可随时清空状态、重新选择语言。

### English

Send any wake word to start the agent and display the interactive language-selection card:

> `hello` · `hi` · `hey` · `你好` · `start` · `help`

Click **🇨🇳 中文** or **🇬🇧 English** — all subsequent bot replies switch to that language.

Type `exit` · `退出` · `reset` at any time to clear state and reselect language.

---

## 2. 视频生成模式 / Video Generation Modes

### 中文

| 模式 | 触发方式 | 底层工具 |
|------|----------|----------|
| **图生视频** | 发图片 → 发描述 | `image_to_video` |
| **文生视频** | 仅发文字描述 | `text_to_video` |
| **多镜头** | 多场景脚本 / "多镜头" | `multi_shot_video` |
| **手办转动漫** | 发手办图片 → 发描述 | `figurine_to_anime → image_to_video` |
| **口播视频** | 图片 + 音频 / 对话文字 | `tts → audio_portrait` |

所有模式均使用火山引擎即梦 AI（`jimeng_ti2v_v30_pro`），默认分辨率 1280×720，时长 4 秒（可设 8 秒）。

### English

| Mode | Trigger | Under the hood |
|------|---------|----------------|
| **Image-to-Video** | Send image → send description | `image_to_video` |
| **Text-to-Video** | Send text description only | `text_to_video` |
| **Multi-Shot** | Multi-scene script / "multi-shot" | `multi_shot_video` |
| **Figurine → Anime** | Send figurine photo → send description | `figurine_to_anime → image_to_video` |
| **Talking Portrait** | Image + audio / dialogue text | `tts → audio_portrait` |

All modes use Volcengine JiMeng AI. Default: 1280×720, 4 s (up to 8 s).

---

## 3. 手办转动漫 / Figurine → Anime

### 中文

上传 Q 版手办照片并发送描述，Agent 自动走两步 Replicate 流水线：

1. **depth-anything-v2-large** — 提取深度图，保留手办的 3D 立体结构
2. **SDXL ControlNet (depth)** — 以深度图为条件，将手办渲染为 2D 动漫风格，保留颜色、服装、发型

转换后的动漫图自动传入 `image_to_video`，生成动画视频。

**所需配置：**
```env
REPLICATE_API_TOKEN=r8_xxx
```

**示例触发：**
```
（发送手办照片）
把这个手办做成动漫动画，背景是江湖酒馆
```

### English

Upload a Q-version figurine photo and send a description. The agent runs a two-step Replicate pipeline:

1. **depth-anything-v2-large** — extracts a depth map, preserving the figurine's 3-D volume
2. **SDXL ControlNet (depth)** — renders in 2-D anime style, preserving colours, outfit, and hair

The anime render is automatically piped into `image_to_video` to produce the final animation.

**Requires:** `REPLICATE_API_TOKEN` in `.env`.

---

## 4. 多镜头生成 / Multi-Shot Generation

### 中文

发送含多个场景的脚本，或在描述中包含"多镜头"、"分镜"、"multi-shot"，Planner 自动选择 `multi_shot_video`。

**工作流程：**
1. DeepSeek 将脚本拆解为 2–4 个独立场景描述
2. 每个场景自动注入专属运镜与光效
3. 所有片段**并行**提交火山引擎，节省等待时间
4. 返回所有片段链接，按镜头编号排列

**示例输入：**
```
场景1：少年骑驴穿越山林，晨雾弥漫
场景2：少年从葫芦里倒出神秘液体
场景3：驴子摇头，两人相视而笑，远山背景
```

**示例输出：**
```
🎬 多镜头视频生成完成！(3 个片段)
  🎞️ 第1镜：https://...
  🎞️ 第2镜：https://...
  🎞️ 第3镜：https://...
```

### English

Send a multi-scene script or use the keywords "multi-shot", "multi-scene", or "分镜". The planner selects `multi_shot_video` automatically.

**Workflow:** DeepSeek splits the script into 2–4 scenes → each gets a unique camera move → all clips submitted to Volcengine **in parallel** → all URLs returned at once.

Individual clip failures are handled gracefully — the remaining clips still deliver.

---

## 5. 口播数字人 / Talking-Head Portrait

### 中文

基于火山引擎 OmniHuman 1.5，将人像照片与音频结合，生成口型同步视频。

**两种触发方式：**

| 方式 | 流程 |
|------|------|
| **自带音频** | 发图片 → 发音频文件 → 发描述 |
| **TTS 驱动** | 发图片 → 发描述（包含人物要说的话）|

TTS 方式由 Agent 自动合成语音（edge-tts），再驱动 OmniHuman，全程无需手动录音。

> ⚠️ 需在火山引擎控制台激活 OmniHuman 1.5 服务。

### English

Powered by Volcengine OmniHuman 1.5. Combines a portrait photo with audio to produce a lip-synced video.

| Mode | Flow |
|------|------|
| **With own audio** | Send image → send audio file → send description |
| **TTS-driven** | Send image → send description with dialogue |

In TTS mode, the agent auto-generates speech via edge-tts and feeds it directly to OmniHuman — no manual recording needed.

---

## 6. 视频配音 / Add Audio to Video

### 中文

视频生成后，回复「**加配音**」并描述想要的声音，Agent 自动识别意图并生成对应音频，无需重新生成视频。

**工作流程：**
1. **LLM 意图分析**：DeepSeek 读取用户的音效需求，判断是「氛围音效」还是「旁白配音」
2. **音频生成（双路径）：**
   - **氛围音效模式**（引擎声、动物声、环境音）→ Replicate `stable-audio-open` 生成 WAV 音效
   - **旁白模式**（配音、旁白、对话）→ `edge-tts` 合成语音旁白
3. `ffmpeg` 将音频循环填充至视频时长后混合
4. 返回带音频的新视频链接

**触发关键词：**
> `加配音` · `加声音` · `加音乐` · `加旁白` · `音效` · `配音` · `add audio` · `add narration`

**示例：**
```
加声音：摩托车引擎声，低沉匀速，带轻微转速起伏；
边车小狗偶尔汪一声，短促软乎。
→ 自动生成引擎+狗叫氛围音，混入视频
```

**所需环境：** `ffmpeg` + `REPLICATE_API_TOKEN`（氛围音效模式）

### English

After receiving a video, reply with "**add audio**" and describe the sound you want. The agent automatically identifies your intent and generates the appropriate audio.

**Pipeline:**
1. **LLM intent analysis**: DeepSeek reads your audio request and decides the mode
2. **Audio generation (two paths):**
   - **Ambient / SFX mode** (engine, animals, environment) → Replicate `stable-audio-open` generates WAV
   - **Narration mode** (voice-over, dialogue) → `edge-tts` synthesises speech
3. `ffmpeg` loops audio to match video duration and merges
4. Returns new video URL with audio

**Trigger keywords:** `加配音` · `add audio` · `add narration` · `sound effects` · `narrate`

**Requires:** `ffmpeg` + `REPLICATE_API_TOKEN` (for ambient SFX mode).

---

## 7. 文字转语音 / Text-to-Speech

### 中文

使用 `edge-tts`（微软免费神经网络 TTS，无需 API Key）合成语音。

| 音色 | 特点 |
|------|------|
| `zh-CN-XiaoxiaoNeural` | 中文女声（默认）|
| `zh-CN-YunxiNeural` | 中文男声 |
| `en-US-JennyNeural` | 英文女声 |
| `en-US-GuyNeural` | 英文男声 |

TTS 在以下场景自动触发：口播视频（无音频文件时）、视频配音（`add_bgm` 工具）。

### English

Uses `edge-tts` (Microsoft free neural TTS, no API key needed) for speech synthesis.

TTS is automatically triggered in two scenarios: talking-head generation (when no audio file is provided) and video narration via the `add_bgm` tool.

---

## 8. 电影级运镜 / Cinematic Camera Language

### 中文

所有视频生成前，用户描述经 DeepSeek 增强为专业电影提示词：

**运镜（每个视频自动选一种）：**
- `slow push-in` 缓推 · `gentle pull-back` 缓拉 · `smooth pan left/right` 横摇
- `overhead crane shot descending` 俯冲航拍 · `low-angle tracking shot` 低角跟拍
- `rack focus` 焦点拉伸 · `static wide shot` 静止全景 · `handheld close-up` 手持特写

**光影（自动匹配场景氛围）：**
- `golden-hour rim light` 黄金时段 · `soft dappled light through leaves` 林间碎光
- `dramatic side-lighting` 戏剧侧光 · `misty volumetric fog` 体积雾
- `moonlit silhouette` 月光剪影 · `warm lantern glow` 灯笼暖光

提示词统一以 `"Hand-drawn animation style, 2D sketch art,"` 开头，保证风格一致。

### English

Before every video generation, user descriptions are enhanced by DeepSeek into professional cinematic prompts with auto-selected camera move and lighting.

**Camera moves:** push-in, pull-back, pan, crane, tracking, rack focus, static wide, handheld close-up.

**Lighting:** golden-hour, dappled forest, dramatic side-lighting, volumetric fog, moonlit silhouette, lantern glow.

All prompts open with `"Hand-drawn animation style, 2D sketch art,"` for visual coherence.

---

## 9. 后期处理框架 / Post-Production Commands

### 中文

视频交付后进入后期模式，支持四类操作，无需重新上传图片：

```
🎬 视频已生成！
🔗 https://...

⭐ 请给这个视频打分（回复 1–5）
1=很差  3=还可以  5=非常满意

💡 也可以回复：加配音 · 重新生成 · 发新图/描述继续创作
```

| 用户回复 | 行为 |
|----------|------|
| `1`–`5` | 打分，记入记忆系统 |
| `加配音` / `配音` | 对当前视频添加旁白，返回带音频版本 |
| `重新生成` / `重做` | 用相同图片+描述重新生成 |
| 其他文字 / 新图片 | 开始全新创作 |

### English

After video delivery, the bot enters post-production mode — no re-upload needed:

| Reply | Action |
|-------|--------|
| `1`–`5` | Rate the video, saved to memory |
| `add audio` / `narrate` | Add narration to the current video |
| `regenerate` / `redo` | Re-run with the same image + prompt |
| Anything else | Start a new creation |

---

## 10. 取消与重置 / Cancel & Reset

### 中文

**取消生成中的任务：**
> `取消` · `停止` · `停` · `不要了` · `算了` · `stop` · `cancel`

**重置全部状态（重新选择语言）：**
> `exit` · `退出` · `重置` · `reset` · `/start`

重置会同时取消正在进行的生成任务，清空所有对话状态，回到初始欢迎卡片。

### English

**Cancel an ongoing generation:**
> `取消` · `停止` · `stop` · `cancel` · `不要了`

**Full reset (re-select language):**
> `exit` · `退出` · `reset` · `/start`

Reset cancels any running task, clears all conversation state, and shows the welcome card again.

---

## 11. 评分反馈 / Rating & Feedback

### 中文

评分（1–5）映射为记忆质量权重，影响记忆优先级：

| 评分 | 质量权重 | 含义 |
|------|----------|------|
| ⭐⭐⭐⭐⭐ 5 | 1.0 | 完全符合预期 |
| ⭐⭐⭐⭐ 4 | 0.8 | 比较满意 |
| ⭐⭐⭐ 3 | 0.6 | 可以接受 |
| ⭐⭐ 2 | 0.2 | 不太满意 |
| ⭐ 1 | 0.1 | 很差 |

低分记忆不会立刻删除，而是在优先级竞争中自然淘汰，保留失败教训供系统学习。

### English

Ratings (1–5) map to quality weights that drive memory priority. Low-rated entries are not deleted immediately — they decay naturally via the priority formula, preserving failure lessons until they're no longer relevant.

---

## 12. 三层记忆系统 / Three-Tier Memory

### 中文

参考操作系统存储层次设计，实现优先级驱逐 + LLM 蒸馏的三层记忆：

```
L1  distilled_style  — 风格摘要（2-3句话），每次 Planner 调用必注入
L2  working[]        — ≤ 20 条工作记忆，优先级排序
L3  archive[]        — 无限量归档，LLM 定期压缩
```

**优先级公式：**
```
priority = 0.35 × freshness + 0.30 × freq_score + 0.35 × quality

freshness  = e^(-λt)               半衰期 24 小时，类 LRU
freq_score = log(n+1) / log(20)    对数平滑，类 LFU
quality    = 用户评分映射（0.1–1.0）
```

**驱逐与蒸馏：**
- L2 满（> 20 条）→ 最低优先级条目移至 L3 归档
- L3 ≥ 30 条 → DeepSeek 压缩为 3-5 条精华，回填 L2
- 每积累 5 个质量 ≥ 0.6 的 L2 条目 → DeepSeek 提炼 `distilled_style`（L1）

**Pin 功能：** 重要记忆可设 `pinned=True`，优先级无限大，永不驱逐。

### English

Inspired by OS memory hierarchy — three tiers with priority eviction and LLM distillation:

```
L1  distilled_style  — 2-3 sentence style summary, injected into every Planner call
L2  working[]        — ≤ 20 entries, priority-sorted
L3  archive[]        — unlimited, LLM-compressed periodically
```

**Priority formula:** `0.35 × freshness + 0.30 × freq_score + 0.35 × quality`

- **Freshness**: exponential decay, 24 h half-life (LRU-like)
- **Frequency**: log-smoothed access count (LFU-like, prevents monopoly)
- **Quality**: user rating (the unique human-feedback signal)

**Eviction:** L2 full → lowest-priority entry → L3. L3 ≥ 30 → LLM compresses to 3-5 insights → back to L2.

**Distillation:** Every 5 high-quality L2 entries trigger a DeepSeek call that synthesises a style summary into L1 — the agent gets progressively smarter with use.

---

## 13. 架构概览 / Architecture Overview

```
用户 (飞书)
    │
    ▼
feishu_bot.py  ──── 语言选择卡片 (POST /feishu/card)
    │                后期处理: 加配音 / 重新生成
    │
    ▼
agent.py  (orchestrator)
    ├── memory.py        L1/L2/L3 三层记忆
    ├── planner.py       DeepSeek → JSON 执行计划
    └── executor.py      逐步执行 + 兜底逻辑
            │
            ▼
        tools.py
        ┌─────────────────────┐
        │ image_to_video      │  火山引擎 JiMeng
        │ text_to_video       │  火山引擎 JiMeng
        │ multi_shot_video    │  火山引擎 JiMeng (并行)
        │ figurine_to_anime   │  Replicate (depth + SDXL)
        │ audio_portrait      │  火山引擎 OmniHuman 1.5
        │ tts                 │  edge-tts (微软免费)
        │ add_bgm             │  DeepSeek 意图分析 → Replicate stable-audio-open / edge-tts + ffmpeg
        └─────────────────────┘

api.py (FastAPI)
    POST /feishu/event   ← 飞书消息 Webhook
    POST /feishu/card    ← 卡片按钮回调
    GET  /media/{file}   ← 本地文件服务（完整响应，避免 EOF）
    POST /animate        ← REST API（可选）
```

**核心依赖 / Core Dependencies:**

| 组件 | 用途 |
|------|------|
| FastAPI + uvicorn | Web 服务层 |
| lark-oapi | 飞书机器人 SDK |
| volcengine | 即梦 AI 视频生成 + OmniHuman |
| openai (DeepSeek) | Planner · Prompt 增强 · 记忆蒸馏 |
| replicate | 手办 → 动漫转换流水线 |
| edge-tts | 免费 TTS（微软神经网络语音）|
| ffmpeg | 视频 + 音频合并（配音功能）|

---

*Built with Volcengine JiMeng AI · DeepSeek · Replicate · edge-tts · ffmpeg*
