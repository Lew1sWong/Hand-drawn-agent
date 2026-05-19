# 手绘动画 AI Agent — 功能介绍
# Hand-Drawn Animation AI Agent — Feature Guide

---

## 目录 / Contents

1. [视频生成 / Video Generation](#1-视频生成--video-generation)
2. [多镜头生成 / Multi-Shot Generation](#2-多镜头生成--multi-shot-generation)
3. [口播数字人 / Talking-Head (OmniHuman)](#3-口播数字人--talking-head-omnihuman)
4. [文字转语音 / Text-to-Speech](#4-文字转语音--text-to-speech)
5. [电影级运镜 / Cinematic Camera Language](#5-电影级运镜--cinematic-camera-language)
6. [飞书机器人 / Feishu Bot](#6-飞书机器人--feishu-bot)
7. [取消指令 / Cancel Command](#7-取消指令--cancel-command)
8. [评分反馈 / Rating & Feedback](#8-评分反馈--rating--feedback)
9. [三层记忆系统 / Three-Tier Memory System](#9-三层记忆系统--three-tier-memory-system)
10. [架构概览 / Architecture Overview](#10-架构概览--architecture-overview)

---

## 1. 视频生成 / Video Generation

### 中文

本 Agent 支持三种视频生成模式：

| 模式 | 触发方式 | 说明 |
|------|----------|------|
| **图生视频** | 发送草图 + 文字描述 | 将手绘草图动画化，最长 8 秒 |
| **文生视频** | 仅发送文字描述 | 无需图片，直接从描述生成视频 |
| **多镜头视频** | 发送含多场景的脚本 | 自动生成 2–4 个独立片段（见第 2 节）|

底层调用火山引擎即梦 AI（`jimeng_ti2v_v30_pro`），分辨率默认 1280×720，时长默认 4 秒（可设置为 8 秒）。

### English

The agent supports three video generation modes:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Image-to-Video** | Send sketch + text description | Animates a hand-drawn sketch, up to 8 seconds |
| **Text-to-Video** | Send text description only | Generates video directly from text, no image needed |
| **Multi-Shot** | Send a multi-scene script | Auto-generates 2–4 independent clips (see Section 2) |

Powered by Volcengine JiMeng AI (`jimeng_ti2v_v30_pro`). Default resolution: 1280×720, default duration: 4 s (configurable up to 8 s).

---

## 2. 多镜头生成 / Multi-Shot Generation

### 中文

当用户发送包含多个场景的脚本（或明确要求"多镜头"、"分镜"、"multi-shot"）时，Planner 自动选择 `multi_shot_video` 工具。

**工作流程：**
1. DeepSeek 将用户脚本拆解为 2–4 个独立场景描述
2. 每个场景自动注入专属运镜语言（推拉摇移、景深、光效）
3. 所有片段**并行**提交至火山引擎，节省等待时间
4. 生成完成后依次返回各镜头链接

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

Pipeline: multi_shot_video
```

### English

When the user sends a detailed multi-scene script, or explicitly requests "multi-shot", "multi-scene", or "分镜", the Planner automatically selects the `multi_shot_video` tool.

**Workflow:**
1. DeepSeek parses the user's script into 2–4 independent scene prompts
2. Each scene gets a unique cinematic camera move injected automatically
3. All clips are submitted to Volcengine **in parallel** to save time
4. All clip URLs are returned once generation completes

**Tip:** You can set `n_shots` (2–4) explicitly in your description. Failed individual clips are skipped gracefully — the rest still deliver.

---

## 3. 口播数字人 / Talking-Head (OmniHuman)

### 中文

基于火山引擎 OmniHuman 1.5，将一张人像照片与音频结合，生成口型同步的说话视频。

**两种使用方式：**
- **自带音频**：发送图片 → 发送音频文件 → 发送描述，直接驱动口播
- **TTS 口播**：发送图片 → 发送描述（包含人物要说的话），Agent 自动合成语音再驱动

**所需服务激活：**
- 火山引擎控制台 → 即梦 AI → OmniHuman 1.5 → 确认状态「已开通」

> ⚠️ 若遇到 504 超时，说明 OmniHuman 服务未完全激活，请联系火山引擎商务开通。

### English

Powered by Volcengine OmniHuman 1.5, combines a portrait photo with audio to produce a lip-synced talking-head video.

**Two usage patterns:**
- **With your own audio:** Send image → send audio file → send description
- **TTS-driven lip sync:** Send image → send description with dialogue text; the agent auto-generates speech and drives the portrait

**Requires:** OmniHuman 1.5 activated in the Volcengine console (即梦 AI → 数字人 → 服务开通).

---

## 4. 文字转语音 / Text-to-Speech

### 中文

使用 `edge-tts`（微软免费 TTS，无需 API Key）将文字合成为音频，再用于口播数字人驱动。

**支持音色：**

| 音色名 | 特点 |
|--------|------|
| `zh-CN-XiaoxiaoNeural` | 女声，普通话，默认 |
| `zh-CN-YunxiNeural` | 男声，普通话 |
| `en-US-JennyNeural` | 英文女声 |

**在 Planner 中的配置：**
```json
{
  "tool": "tts",
  "inputs": {
    "tts_text": "江湖再见，小白！",
    "voice": "zh-CN-XiaoxiaoNeural"
  }
}
```

### English

Uses `edge-tts` (Microsoft free TTS, no API key required) to synthesize speech, which then drives the OmniHuman portrait animation.

The TTS tool is automatically planned before `audio_portrait` when the user wants the character to speak but provides no audio file. The generated `.mp3` is served via the `/media/` endpoint and consumed directly by the portrait tool.

---

## 5. 电影级运镜 / Cinematic Camera Language

### 中文

所有视频生成前，用户描述都经过 DeepSeek 增强为专业的电影提示词，包含：

**运镜类型（每个视频自动选择一种）：**
- `slow push-in` 缓慢推进
- `gentle pull-back` 缓慢拉远
- `smooth pan left/right` 横向摇镜
- `overhead crane shot descending` 俯冲航拍
- `low-angle tracking shot` 低角度跟拍
- `rack focus` 焦点拉伸（前景→背景）
- `static wide shot` 静止全景

**光影类型（自动匹配场景氛围）：**
- `golden-hour rim light` 黄金时段轮廓光
- `soft dappled light through leaves` 林间碎光
- `dramatic side-lighting` 戏剧性侧光
- `misty volumetric fog` 雾气氛围
- `moonlit silhouette` 月光剪影
- `warm lantern glow` 暖色灯笼光

所有提示词以 `"Hand-drawn animation style, 2D sketch art,"` 开头，以 `"consistent line-art aesthetic, fluid animation."` 结尾，保证风格一致。

### English

Before every video generation, user descriptions are enhanced by DeepSeek into professional cinematic prompts, including:

**Auto-selected camera move** (one per video):
push-in, pull-back, pan, crane, tracking shot, rack focus, or static wide.

**Auto-matched lighting** (based on scene mood):
golden-hour rim light, dappled forest light, dramatic side-lighting, volumetric fog, moonlit silhouette, or warm lantern glow.

All prompts are framed as `"Hand-drawn animation style, 2D sketch art, …, consistent line-art aesthetic, fluid animation."` to guarantee visual coherence across shots.

---

## 6. 飞书机器人 / Feishu Bot

### 中文

通过飞书（Lark）即可使用全部功能，无需命令行。

**对话流程：**

```
用户                          Bot
 |                             |
 |── 发送图片 ──────────────▶ ✅ 收到草图，请发描述（或先发音频）
 |                             |
 |── (可选) 发送音频 ────────▶ 🎵 收到音频，请发描述
 |                             |
 |── 发送文字描述 ──────────▶ ⏳ 生成中... (每30秒进度提醒)
 |                             |
 |                            🎬 视频链接
 |                             |
 |                            ⭐ 请打分 1-5
 |                             |
 |── 回复数字 ───────────────▶ 谢谢反馈！
```

**支持消息类型：**
- `image` — 草图/照片
- `audio` / `file` — 音频驱动口播
- `text` — 描述文字
- `post` — 飞书富文本（多行格式化文本，如脚本）

**环境变量配置：**
```env
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=xxx
FEISHU_VERIFICATION_TOKEN=xxx
PUBLIC_BASE_URL=https://your-server.com
```

### English

All features are accessible via the Feishu (Lark) bot — no CLI required.

**Supported message types:** image, audio/file, plain text, and rich-text "post" messages (multi-line scripts).

**Progress updates** are sent every 30 seconds during generation. **Multi-shot** results display all clip URLs numbered by shot.

**Setup:** Create an Enterprise Internal App in open.feishu.cn, grant `im:message:send_as_bot` + `im:resource` scopes, subscribe to `im.message.receive_v1`, and set the webhook URL to `https://your-server.com/feishu/event`.

---

## 7. 取消指令 / Cancel Command

### 中文

在视频生成过程中，随时发送以下任意关键词可立即中止任务：

> `取消` `停止` `停` `不要了` `算了` `stop` `cancel`

中止后状态重置，可立即开始新的创作。

### English

During generation, send any of the stop keywords to cancel immediately:

> `取消` `停止` `停` `不要了` `算了` `stop` `cancel`

The task is cancelled via `asyncio.Task.cancel()`, state resets to idle, and you can start a new request right away.

---

## 8. 评分反馈 / Rating & Feedback

### 中文

每次视频生成完成后，Bot 会询问：

```
⭐ 请给这个视频打分（回复 1–5）
1=很差  3=还可以  5=非常满意
```

评分映射到 Memory 质量权重：

| 评分 | 质量权重 | 含义 |
|------|----------|------|
| 5 ⭐ | 1.0 | 完全符合预期 |
| 4 ⭐ | 0.8 | 比较满意 |
| 3 ⭐ | 0.6 | 可以接受 |
| 2 ⭐ | 0.2 | 不太满意 |
| 1 ⭐ | 0.1 | 很差 |
| 未评分 | 0.5 | 中性默认值 |

> 低分记忆**不会立刻删除**，而是在优先级竞争中自然淘汰，保留失败教训。

### English

After each video delivery, the bot asks for a 1–5 star rating. Ratings are converted to a quality weight (0.1–1.0) stored in the memory entry. Low-rated entries are not deleted immediately — they compete fairly in the priority queue and are naturally evicted over time, preserving the "failure lesson" until it's no longer needed.

---

## 9. 三层记忆系统 / Three-Tier Memory System

### 中文

借鉴操作系统存储层次设计，实现了一套优先级驱逐 + LLM 蒸馏的记忆系统：

```
L1  distilled_style   — 高密度风格摘要，始终注入 Planner Prompt
L2  working[]         — ≤ 20 条工作记忆，优先级排序
L3  archive[]         — 无限量归档，LLM 定期压缩
```

**优先级公式：**

```
priority = 0.35 × freshness(t) + 0.30 × freq_score + 0.35 × quality

freshness = e^(-λt)         λ = ln(2)/24h，半衰期 24 小时
freq_score = log(n+1)/log(20)  log 平滑，避免高频条目垄断
quality    = 用户评分映射值（0.1–1.0）
```

**驱逐流程（类比 OS 页面置换）：**

```
L2 满（>20条）
    ↓
找 min(priority) 的未锁定条目
    ↓
移至 L3 Archive
    ↓  L3 ≥ 30 条
LLM 压缩为 3-5 条精华
    ↓
回填 L2（promote）
```

**L1 蒸馏触发：**
每积累 5 个质量 ≥ 0.6 的工作记忆，DeepSeek 自动提炼用户视觉风格偏好，更新 `distilled_style`，直接注入未来所有 Planner 调用，越用越精准。

**Pin 功能（类比 OS mlock）：**
重要记忆可设为 `pinned=True`，优先级无限大，永不被驱逐。

### English

Inspired by OS memory hierarchy, the agent implements a three-tier priority-managed memory system:

```
L1  distilled_style   — compact style summary, always injected into Planner
L2  working[]         — ≤ 20 MemoryEntry items, priority-sorted
L3  archive[]         — unlimited, LLM-compressed periodically
```

**Priority formula:**
```
priority = 0.35 × freshness + 0.30 × freq_score + 0.35 × quality
```

- **Freshness** decays exponentially (half-life = 24 h) — like LRU
- **Frequency** is log-smoothed (like LFU, avoids monopoly)
- **Quality** comes from user ratings — the unique human feedback signal

**Eviction:** When L2 is full, the lowest-priority unpinned entry is moved to L3. When L3 reaches 30 entries, LLM compresses the archive into 3–5 high-value insight entries and promotes them back to L2.

**Distillation:** Every 5 high-quality (≥ 0.6) L2 entries trigger a DeepSeek call that synthesises a 2–3 sentence style summary (`distilled_style`). This is injected into every future Planner prompt as L1 — making the agent progressively smarter the more you use it.

**Pin (like OS mlock):** Critical entries can be pinned (`pinned=True`) — their priority is `∞` and they are never evicted.

---

## 10. 架构概览 / Architecture Overview

```
用户 (飞书) ──▶ feishu_bot.py ──▶ agent.py
                    │                  │
                    │              planner.py (DeepSeek)
                    │                  │
                    │              executor.py
                    │                  │
                    │         ┌────────┴────────────────┐
                    │     tools.py                  memory.py
                    │   ┌──────────────┐          ┌──────────────┐
                    │   │ image_to_video│          │ L1 distilled │
                    │   │ text_to_video │          │ L2 working[] │
                    │   │ multi_shot   │          │ L3 archive[] │
                    │   │ tts          │          └──────────────┘
                    │   │ audio_portrait│
                    │   └──────────────┘
                    │
              api.py (FastAPI)
              /feishu/event  ← 火山引擎 Webhook
              /media/{file}  ← TTS & 飞书下载文件服务
```

**核心依赖 / Core Dependencies:**

| 组件 | 用途 |
|------|------|
| FastAPI + uvicorn | Web 服务层 |
| lark-oapi | 飞书 SDK |
| volcengine | 即梦 AI 视频生成 |
| openai (DeepSeek) | Planner + Prompt 增强 + 记忆蒸馏 |
| edge-tts | 免费 TTS（微软神经网络语音）|

---

*Built with ❤️ using Volcengine JiMeng AI + DeepSeek + edge-tts*
