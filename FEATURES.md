# 手绘动画 AI Agent — 功能全览
# Hand-Drawn Animation AI Agent — Complete Feature Guide

---

## 目录 / Contents

0. [本地启动 / Local Startup](#0-本地启动--local-startup)
1. [唤醒与语言选择 / Wake & Language](#1-唤醒与语言选择--wake--language)
2. [视频生成模式 / Video Generation Modes](#2-视频生成模式--video-generation-modes)
3. [手办转动漫 / Figurine → Anime](#3-手办转动漫--figurine--anime)
4. [多镜头生成 / Multi-Shot Generation](#4-多镜头生成--multi-shot-generation)
5. [口播数字人 / Talking-Head Portrait](#5-口播数字人--talking-head-portrait)
6. [视频配音 / Add Audio to Video](#6-视频配音--add-audio-to-video)
7. [文字转语音 / Text-to-Speech](#7-文字转语音--text-to-speech)
8. [电影级运镜 / Cinematic Camera Language](#8-电影级运镜--cinematic-camera-language)
9. [后期处理 / Post-Production Commands](#9-后期处理--post-production-commands)
10. [取消与重置 / Cancel & Reset](#10-取消与重置--cancel--reset)
11. [评分反馈 / Rating & Feedback](#11-评分反馈--rating--feedback)
12. [三层记忆系统 / Three-Tier Memory](#12-三层记忆系统--three-tier-memory)
13. [架构概览 / Architecture Overview](#13-架构概览--architecture-overview)

---

## 0. 本地启动 / Local Startup

### 前置条件

1. **安装依赖**（首次）
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量** — 复制并填写 `.env`：
   ```bash
   cp .env.example .env
   ```
   必填项：

   | 变量 | 说明 | 获取地址 |
   |------|------|----------|
   | `VOLC_ACCESSKEY` / `VOLC_SECRETKEY` | 火山引擎（即梦 AI 视频生成） | console.volcengine.com → IAM → 密钥管理 |
   | `DEEPSEEK_API_KEY` | DeepSeek LLM | platform.deepseek.com → API Keys |
   | `REPLICATE_API_TOKEN` | 手办→动漫 + 音效生成 | replicate.com → Account → API Tokens |
   | `FREESOUND_API_KEY` | 免费音效库（可选） | freesound.org/apiv2/apply |
   | `FEISHU_APP_ID` / `FEISHU_APP_SECRET` / `FEISHU_VERIFICATION_TOKEN` | 飞书机器人 | open.feishu.cn → 应用凭证 |
   | `PUBLIC_BASE_URL` | 外网可访问的服务地址 | 见下方 localtunnel 配置 |

### 启动步骤（每次开发）

需要**同时**开两个终端窗口：

**终端 1 — FastAPI 服务**
```bash
cd /path/to/hand_drawn_agent
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**终端 2 — localtunnel 内网穿透**
```bash
npx localtunnel --port 8000 --subdomain angry-adults-read
```

启动后访问：
- **API 服务：** http://localhost:8000
- **Swagger UI：** http://localhost:8000/docs
- **外网地址：** https://angry-adults-read.loca.lt

停止服务：`Ctrl+C`（前台）或 `kill $(lsof -ti:8000)`（后台）

### 飞书机器人配置（仅首次）

1. 进入 [open.feishu.cn](https://open.feishu.cn) → **开发者后台** → 创建企业自建应用
2. **权限管理** → 开通：`im:message:send_as_bot`、`im:resource`
3. **事件订阅** → 添加事件 `im.message.receive_v1`
   - 请求 URL：`https://angry-adults-read.loca.lt/feishu/event`
4. **机器人** → 卡片回调地址：`https://angry-adults-read.loca.lt/feishu/card`
5. 将 App ID / App Secret / Verification Token 填入 `.env`

---

## 1. 唤醒与语言选择 / Wake & Language

### 中文

发送任意唤醒词启动 Agent，弹出交互式语言选择卡片：

> `hello` · `hi` · `你好` · `嗨` · `开始` · `start` · `help` · `帮助`

卡片提供两个按钮：**🇨🇳 中文** / **🇬🇧 English**，点击后所有后续回复切换为对应语言。

发送 `exit` · `退出` · `重置` · `reset` · `/start` 可随时清空状态、重新选择语言。

跳过卡片直接发图片或描述也可使用，默认语言为中文。

### English

Send any wake word to display the interactive language-selection card:

> `hello` · `hi` · `hey` · `你好` · `start` · `help`

Click **🇨🇳 中文** or **🇬🇧 English** — all subsequent bot replies switch to that language.

Type `exit` · `退出` · `reset` · `/start` at any time to clear state and reselect language. Skipping onboarding is fine — the bot defaults to Chinese.

---

## 2. 视频生成模式 / Video Generation Modes

### 中文

| 模式 | 触发方式 | 底层工具 |
|------|----------|----------|
| **图生视频** | 发图片 → 发描述 | `image_to_video` |
| **文生视频** | 仅发文字描述 | `text_to_video` |
| **多镜头** | 多场景脚本 / "多镜头" / "分镜" | `multi_shot_video` |
| **手办转动漫** | 发手办图片 + 描述 | `figurine_to_anime → image_to_video` |
| **口播视频** | 图片 + 音频文件 | `audio_portrait` |
| **TTS 口播** | 图片 + 含对白的描述 | `tts → audio_portrait` |

所有视频使用火山引擎即梦 AI（`jimeng_ti2v_v30_pro`），默认分辨率 1280×720，时长 4 秒（可设 8 秒）。

**Planner 决策机制（三层）：**
1. **规则层**（无 LLM）：有图片、有手办关键词、有多镜头词、有图+音频 → 直接构建计划，跳过 LLM
2. **LLM 层**（DeepSeek）：仅用于纯文字请求等模糊情况
3. **校验层**：移除幻觉工具名，将规则层的音频分类注入 LLM 生成的计划

### English

| Mode | Trigger | Under the hood |
|------|---------|----------------|
| **Image-to-Video** | Send image → send description | `image_to_video` |
| **Text-to-Video** | Text description only | `text_to_video` |
| **Multi-Shot** | Multi-scene script / "multi-shot" / "分镜" | `multi_shot_video` |
| **Figurine → Anime** | Figurine photo + description | `figurine_to_anime → image_to_video` |
| **Talking Portrait** | Image + audio file | `audio_portrait` |
| **TTS Portrait** | Image + description with dialogue | `tts → audio_portrait` |

All video modes use Volcengine JiMeng AI. Default: 1280×720, 4 s (up to 8 s).

**Three-layer Planner:** deterministic rules handle unambiguous cases without any LLM call; DeepSeek is only invoked for genuinely ambiguous inputs (e.g. pure text with no keywords).

---

## 3. 手办转动漫 / Figurine → Anime

### 中文

上传 Q 版手办照片并发送描述，Agent 走两步 Replicate 流水线：

1. **depth-anything-v2-large** — 提取深度图，保留手办的 3D 立体结构
2. **SDXL ControlNet (depth)** — 以深度图为条件，渲染为 2D 动漫风格，保留颜色、服装、发型

渲染后的动漫图自动传入 `image_to_video` 生成动画。

**所需配置：**
```env
REPLICATE_API_TOKEN=r8_xxx
```

**触发关键词：** `手办` · `figurine` · `figure` · `toy` · `q版`（需同时发送图片）

**示例：**
```
（上传手办照片）
把这个手办做成动漫动画，背景是江湖酒馆
```

### English

Upload a Q-version figurine photo with a description. The agent runs a two-step Replicate pipeline:

1. **depth-anything-v2-large** — depth map extraction, preserving 3-D chibi volume
2. **SDXL ControlNet (depth)** — 2-D anime render preserving colours, outfit, and hair

The anime render is automatically piped into `image_to_video`.

**Requires:** `REPLICATE_API_TOKEN` in `.env`.

**Trigger keywords:** `手办` · `figurine` · `figure` · `toy` · `q版` (with an uploaded image).

---

## 4. 多镜头生成 / Multi-Shot Generation

### 中文

发送含多个场景的脚本，或描述中包含"多镜头"、"分镜"、"multi-shot"，Planner 规则层直接选择 `multi_shot_video`（不调 LLM）。

**工作流程：**
1. DeepSeek 将脚本拆解为 2–4 个独立场景描述，每个注入专属运镜与光效
2. 所有片段**并行**提交火山引擎
3. 返回所有片段链接，按镜头编号排列
4. 单个片段失败不影响其余片段交付

**示例输入：**
```
场景1：少年骑驴穿越山林，晨雾弥漫
场景2：少年从葫芦里倒出神秘液体
场景3：驴子摇头，两人相视而笑，远山背景
```

**示例输出：**
```
🎬 多镜头视频生成完成！(3 个片段)
  🎞️ 第1镜：https://…
  🎞️ 第2镜：https://…
  🎞️ 第3镜：https://…
```

### English

Include "多镜头", "分镜", or "multi-shot" in your description, or send a multi-scene script. The rule layer picks `multi_shot_video` directly without calling DeepSeek.

**Workflow:** DeepSeek splits the script into 2–4 prompts → each gets a unique camera move and lighting → all clips submitted to Volcengine **in parallel** → all URLs returned at once. Individual clip failures are graceful — remaining clips still deliver.

---

## 5. 口播数字人 / Talking-Head Portrait

### 中文

基于火山引擎 OmniHuman 1.5，将人像照片与音频结合，生成口型同步视频。

**两种触发方式：**

| 方式 | 流程 |
|------|------|
| **自带音频** | 发图片 → 发音频文件 → 发描述 → `audio_portrait` |
| **TTS 驱动** | 发图片 → 发含对白的描述 → `tts → audio_portrait` |

TTS 方式由 Agent 自动合成语音（edge-tts），无需手动录音。

> ⚠️ 需在火山引擎控制台激活 OmniHuman 1.5 服务（即梦AI → 数字人 → 服务开通）。

### English

Powered by Volcengine OmniHuman 1.5. Combines a portrait photo with audio to produce a lip-synced video.

| Mode | Flow |
|------|------|
| **With own audio** | Image → audio file → description → `audio_portrait` |
| **TTS-driven** | Image → description with dialogue → `tts → audio_portrait` |

In TTS mode the agent auto-generates speech via edge-tts and feeds it directly to OmniHuman.

> ⚠️ OmniHuman 1.5 must be activated in the Volcengine console before use.

---

## 6. 视频配音 / Add Audio to Video

### 中文

视频交付后，回复"**加配音**"（或其他触发词）并描述想要的声音，Agent 自动路由到正确的音频引擎。

**意图路由（两层决策）：**

| 关键词 | 分类 | 引擎 |
|--------|------|------|
| 背景音、音效、环境音、引擎、风声、雨声、鸟叫、music | **SFX** | Replicate `stable-audio-open` → WAV |
| 旁白、朗读、让他说、让她说、说出 | **旁白** | `edge-tts` → MP3 |
| 声音、配音、音乐、sound、audio（模糊） | **SFX（默认）** | Replicate `stable-audio-open` |

**决策流程：**
1. **规则层**：关键词命中 → 直接注入 `mode=sfx` 或 `mode=narration`，`add_bgm` 跳过 LLM
2. **LLM 层**（仅模糊情况）：DeepSeek 分析意图，默认输出 `ambient`，绝不默认 `narration`
3. `ffmpeg` 将音频循环填充至视频时长后混合
4. 返回带音频的新视频链接

**触发关键词：**
> `加配音` · `加声音` · `加音乐` · `加旁白` · `音效` · `add audio` · `add narration` · `add sound`

**示例（SFX）：**
```
加声音：摩托车引擎声，低沉匀速，带轻微转速起伏；
边车小狗偶尔汪一声，短促软乎。
→ 规则层识别「引擎」→ mode=sfx → stable-audio-open 生成引擎+狗叫氛围音
```

**所需环境：** `ffmpeg` + `REPLICATE_API_TOKEN`（SFX 模式）

### English

After receiving a video, reply with "**add audio**" and describe the sound. Intent is routed deterministically before any LLM call.

**Routing logic:**

| Keywords | Mode | Engine |
|----------|------|--------|
| ambient, sfx, sound effect, 音效, engine, wind, rain, 背景音 | **SFX** | Replicate `stable-audio-open` → WAV |
| narrate, voice over, 旁白, 朗读, 让他说 | **Narration** | `edge-tts` → MP3 |
| 声音, 配音, sound, audio (ambiguous) | **SFX (default)** | Replicate `stable-audio-open` |

Confident matches bypass the `_plan_audio` LLM call entirely. Ambiguous inputs fall back to DeepSeek, which is also instructed to default to `ambient`, never `narration`.

`ffmpeg` loops audio to match video duration and merges. Returns new video URL.

**Requires:** `ffmpeg` + `REPLICATE_API_TOKEN` (for SFX mode).

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

TTS 在两个场景自动触发：
- **口播视频**（无音频文件时）：`tts → audio_portrait`
- **视频配音旁白模式**：`add_bgm` 内部调用

### English

Uses `edge-tts` (Microsoft free neural TTS, no API key needed).

Automatically triggered in two scenarios:
- **Talking-head generation** (no audio file uploaded): `tts → audio_portrait`
- **Video narration** via `add_bgm` in narration mode

---

## 8. 电影级运镜 / Cinematic Camera Language

### 中文

所有视频生成前，用户描述经 DeepSeek 增强为专业电影提示词。

**运镜（每个视频自动选一种）：**
- `slow push-in` 缓推 · `gentle pull-back` 缓拉 · `smooth pan left/right` 横摇
- `overhead crane shot descending` 俯冲航拍 · `low-angle tracking shot` 低角跟拍
- `rack focus` 焦点拉伸 · `static wide shot` 静止全景 · `handheld close-up` 手持特写 · `360 orbit` 环绕

**光影（自动匹配场景氛围）：**
- `golden-hour rim light` 黄金时段 · `soft dappled light through leaves` 林间碎光
- `dramatic side-lighting` 戏剧侧光 · `misty volumetric fog` 体积雾
- `moonlit silhouette` 月光剪影 · `warm lantern glow` 灯笼暖光 · `cool blue-tinted dawn` 冷蓝晨光

提示词统一以 `"Hand-drawn animation style, 2D sketch art,"` 开头，保证风格一致。增强仅在 `image_to_video` / `text_to_video` 步骤触发，且在同一次运行中只增强一次。

### English

Before every video generation, user descriptions are enhanced by DeepSeek into cinematic prompts. One camera move and one lighting style are automatically selected per video.

All prompts open with `"Hand-drawn animation style, 2D sketch art,"`. Enhancement runs lazily — only for tools that need it (`image_to_video`, `text_to_video`), and only once per pipeline run.

---

## 9. 后期处理 / Post-Production Commands

### 中文

视频交付后进入等待评分状态，支持以下操作，无需重新上传图片：

```
🎬 视频已生成！
🔗 https://…

⭐ 请给这个视频打分（回复 1–5）
1=很差  3=还可以  5=非常满意

💡 也可以回复：加配音 · 重新生成 · 发新图/描述继续创作
```

| 用户回复 | 行为 |
|----------|------|
| `1`–`5` | 打分并记入记忆系统，切换到待图状态 |
| `加配音` / `add audio` 等 | 对当前视频添加音频，返回带音频版本，继续留在评分状态 |
| `重新生成` / `redo` / `regenerate` | 用相同图片+描述重新生成，返回新视频 |
| 其他文字 / 新图片 | 跳过评分，开始全新创作 |

进度通知：生成期间每 30 秒自动推送一次进度消息，显示已等待时长。

### English

After video delivery, the bot enters post-production mode — no re-upload needed. A 30-second progress ping fires during generation to show elapsed time.

| Reply | Action |
|-------|--------|
| `1`–`5` | Rate the video → saved to memory → return to idle |
| `add audio` / `加配音` etc. | Add audio to current video → return audio+video → stay in rating state |
| `regenerate` / `redo` / `重新生成` | Re-run with same image + prompt → new video |
| Anything else / new image | Skip rating → start new creation |

---

## 10. 取消与重置 / Cancel & Reset

### 中文

**取消生成中的任务：**
> `取消` · `停止` · `停` · `不要了` · `算了` · `stop` · `cancel`

**重置全部状态（重新选择语言）：**
> `exit` · `退出` · `重置` · `reset` · `/start` · `/exit` · `/reset`

重置会同时取消正在进行的生成任务，清空所有对话状态，回到初始欢迎卡片。

### English

**Cancel an ongoing generation:**
> `取消` · `停止` · `stop` · `cancel` · `不要了` · `算了`

**Full reset (re-select language):**
> `exit` · `退出` · `reset` · `/start` · `/exit` · `/reset`

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
| 未评分 | 0.5 | 中性（默认）|
| ⭐⭐ 2 | 0.2 | 不太满意 |
| ⭐ 1 | 0.1 | 很差 |

低分记忆不会立刻删除，而是在优先级竞争中自然衰减，保留失败教训供系统参考。

### English

Ratings (1–5) map to quality weights that drive memory priority. Low-rated entries are not deleted immediately — they decay naturally via the priority formula, preserving failure lessons until they age out.

---

## 12. 三层记忆系统 / Three-Tier Memory

### 中文

参考操作系统存储层次设计，实现优先级驱逐 + LLM 蒸馏的三层记忆（`memory.py`）：

```
L1  distilled_style  — 风格摘要（2-3句话），每次 Planner 调用必注入
L2  working[]        — ≤ 20 条工作记忆，优先级排序
L3  archive[]        — 无限量归档，LLM 定期压缩
```

**优先级公式：**
```
priority = 0.35 × freshness + 0.30 × freq_score + 0.35 × quality

freshness  = e^(-λt)               半衰期 24 小时（类 LRU）
freq_score = log(n+1) / log(20)    对数平滑，访问越频繁越高（类 LFU）
quality    = 用户评分映射（0.1–1.0）
```

**驱逐与蒸馏（后台异步，不阻塞生成）：**
- L2 满（> 20 条）→ 最低优先级条目移至 L3 归档
- L3 ≥ 30 条 → DeepSeek 压缩为 3–5 条精华，回填 L2
- 每积累 5 个质量 ≥ 0.6 的 L2 条目 → DeepSeek 提炼风格摘要写入 L1

**Pin 功能：** `pinned=True` 的记忆优先级无限大，永不驱逐。

### English

Inspired by OS memory hierarchy — three tiers with priority eviction and LLM distillation, all in `memory.py`:

```
L1  distilled_style  — 2-3 sentence style summary, injected into every Planner call
L2  working[]        — ≤ 20 entries, priority-sorted
L3  archive[]        — unlimited, LLM-compressed periodically
```

**Priority formula:** `0.35 × freshness + 0.30 × freq_score + 0.35 × quality`

- **Freshness**: exponential decay, 24 h half-life (LRU-like)
- **Frequency**: log-smoothed access count (LFU-like)
- **Quality**: user rating (0.1–1.0)

**Eviction & distillation run asynchronously** after each generation — they never block video delivery.

---

## 13. 架构概览 / Architecture Overview

```
用户 (飞书 / Telegram / REST)
    │
    ▼
bots/feishu_bot.py  ──── 语言卡片 / 状态机 / 后期处理
bots/telegram_bot.py ─── Telegram 适配层
api.py (FastAPI)
    POST /feishu/event     ← 飞书消息 Webhook
    POST /feishu/card      ← 卡片按钮回调
    POST /animate          ← REST API 提交任务 (202 + job_id)
    GET  /animate/{job_id} ← 轮询任务状态
    GET  /media/{file}     ← 本地文件服务
    GET  /health           ← 健康检查
    │
    ▼
agent.py  (orchestrator)
    ├── memory.py               L1/L2/L3 三层记忆
    ├── planner/                三层规划器
    │   ├── rules.py            规则层（确定性分类，无 LLM）
    │   ├── prompt.py           LLM 系统提示词构建
    │   └── planner.py          rules → LLM → validate+patch
    └── executor.py             逐步执行 + 提示增强 + 兜底逻辑
            │
            ▼
        tools/
        ├── base.py             BaseTool + ToolContract（ctx 依赖自描述）
        ├── _volcengine.py      共享 submit/poll helpers
        ├── _replicate.py       共享 Replicate URL 解析
        ├── registry.py         ALL_TOOLS, TOOL_MAP
        ├── video/
        │   ├── image_to_video.py   火山引擎 JiMeng 3.0 Pro
        │   ├── text_to_video.py    火山引擎 JiMeng 3.0 Pro
        │   ├── multi_shot.py       并行多片段生成
        │   └── figurine.py         Replicate depth + SDXL ControlNet
        └── audio/
            ├── tts.py              edge-tts（微软免费神经网络）
            ├── audio_portrait.py   火山引擎 OmniHuman 1.5
            └── add_bgm.py          规则路由 → stable-audio-open / edge-tts + ffmpeg
```

**核心依赖 / Core Dependencies:**

| 组件 | 用途 |
|------|------|
| FastAPI + uvicorn | Web 服务层 |
| lark-oapi | 飞书机器人 SDK |
| volcengine | 即梦 AI 视频生成 + OmniHuman 1.5 |
| openai (DeepSeek) | Planner LLM · 提示增强 · 场景拆解 · 记忆蒸馏 |
| replicate | 手办→动漫 (depth-anything + SDXL) + stable-audio-open |
| edge-tts | 免费 TTS（微软神经网络语音）|
| ffmpeg | 视频 + 音频合并（配音功能）|
| aiogram | Telegram 机器人 SDK |

---

*Built with Volcengine JiMeng AI · OmniHuman 1.5 · DeepSeek · Replicate · edge-tts · ffmpeg*
