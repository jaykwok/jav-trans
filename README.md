# jav-trans

jav-trans 是一个面向 Windows + NVIDIA 显卡的本地 JAV 字幕生成工具。它把视频处理成日文字幕、中文字幕或中日双语字幕，并把音频准备、切分、Qwen ASR、CTC 强制对齐、字幕时间轴、LLM 翻译和质量报告串成一条本地优先的流水线。

项目目标：本地完成视频、音频、切分、ASR 和字幕时间轴重计算；LLM 只负责翻译、术语一致和口吻连贯，不负责脑补剧情或修正 ASR 误听。

**翻译可以完全本地运行**：除 OpenAI 兼容 API 外，内置 llama.cpp 后端托管 GGUF 量化模型（适配 8G 显存，无需选型）。选本地后端时整条流水线不出网。详见下方「翻译后端支持」。

致谢：[WhisperJAV](https://github.com/a63n/WhisperJAV) 为本项目早期路线提供了重要参考。

---

## 界面预览

![网页控制台主界面](docs/images/ui-web-console.png)

任务提交、翻译后端选择（OpenAI 兼容 API / 本地 Hy-MT2）、实时阶段进度、显存与耗时监控、质量报告都在本地网页控制台完成。更多截图放在 `docs/images/`。

---

## 设计原则

本项目在 ASR 之前**不做任何丢弃式判断**。音频只被切开，不被筛选：切点由 ASR encoder 上的一个 CTC 对齐头给出的 blank 游程决定，输出精确铺满整个文件，每一秒都会进入解码器。

这条原则来自代价不对称：**丢错不可逆，留错可过滤**。判断放在有文本之后。

当前职责划分：

- **切分**只决定边界落在哪里。切点由对齐头的停顿给出；没有配对齐头时退化为定长切分。切错只是边界变差，永远丢不了词。
- **ASR 解码**输出日文文本。
- **CTC 强制对齐**把文本逐字对回音频，产出真实字级时间戳与对齐分数。
- **后置闸**（`src/asr/postgate.py`）只对已有文本**打标不删除**：失控重复、不可能的语速、与邻块重复等。标记写进 ASR 产物和 segment，**目前没有任何阶段按它过滤**——它是可观测性，不是过滤器（`min_alignment_score` 未标定，那一项默认关闭）。
- **字幕 layout** 只在完整的实测字/词时间轴上选择安全边界，不反向修改 ASR chunk 语义，也不为了版面目标伪造时间。

设计演进、实验记录和失败路线见 [docs/HISTORY.md](docs/HISTORY.md)。

---

## 快速开始

### 从Releases下载并运行

解压发布包，双击 `jav-trans.exe`。它自带 FFmpeg Shared（uv 在首次运行时下载到程序目录的 `bin/`），第一次运行会：

1. 读取 PyTorch 官方源上的 torch 安装包测速，报出实测速度和预计耗时；太慢或连不上时可以在控制台填写本地代理，代理会写入 `.env`，之后下载 ASR 模型也复用同一份设置。

   例如设置代理地址127.0.0.1，端口设置为7890，代理协议为http等。

2. 用 uv 同步依赖，控制台实时显示下载进度。安装后约占 3.3GB 磁盘。中断后重新双击即可续装。

装完自动启动，程序窗口出现后控制台窗口自动隐藏；之后每次双击同一个 `jav-trans.exe` 会跳过安装直接启动（它既是安装器也是启动器）。ASR 模型（约 3.9GB）和 CTC 对齐头在第一次转录时按需下载到 `models/`。

每次启动都会先做一次结构自检（程序文件、FFmpeg 共享库、`.venv` 里的关键包）。安装记录只说明「装过」，删掉几个包不会改变它，所以缺包时会显示「修复运行环境」并只补回缺的那几个，而不是整个重装。窗口始终打不开时运行 `jav-trans.exe --doctor`：它会真的导入一遍 torch 等库，报出缺什么、补装、再启动；启动失败时控制台不会隐藏，也会提示运行 `--doctor`。

发布包不打包 PyTorch 和模型权重，因此需要解压到有 15GB 以上空闲空间、且不需要管理员权限的目录（`.venv`、`models/`、`tmp/`、uv 缓存都建在程序目录内，不会写注册表或改系统环境变量）。可用参数：`--doctor` 自检并修复，`--keep-console` 窗口打开后保留控制台，`--proxy <URL>` 指定代理，`--yes` 不提问，`--reinstall` 重装环境，`--install-only` 只装不启动。

用 API 翻译前先在界面「翻译设置」里填好 API Key 并保存；漏填就开始任务会被当场拦下并说明缺什么，只要日文字幕可以打开「不翻译（仅日文字幕）」。

### 从源码运行

推荐环境：

- Windows 10/11。
- NVIDIA 独立显卡和较新的驱动。
- Python 3.14+（与 `pyproject.toml` 的运行时约束一致）。
- FFmpeg Shared（TorchCodec 需要 FFmpeg 共享 DLL），并确保命令行能直接执行 `ffmpeg`。
- Git。

Windows 请安装 Shared 版。不要同时保留会优先占用 `PATH` 的 `Gyan.FFmpeg`
静态版：

```powershell
winget uninstall --id Gyan.FFmpeg --exact
winget install --id Gyan.FFmpeg.Shared --exact

# 关闭并重新打开终端后验证：应指向 full_build-shared\bin
where.exe ffmpeg
Get-ChildItem (Split-Path (Get-Command ffmpeg).Source) -Filter "avcodec-*.dll"
```

如果 `where.exe ffmpeg` 仍列出多个版本，请确保
`Gyan.FFmpeg.Shared\...\full_build-shared\bin` 排在 `PATH` 最前，随后重启
jav-trans。仅有 `ffmpeg.exe` 还不够；其目录必须同时存在
`avcodec-*.dll`、`avformat-*.dll` 和 `avutil-*.dll`。

项目安装：

```powershell
git clone https://github.com/jaykwok/jav-trans.git
cd jav-trans

uv venv
uv sync
```

Qwen3-ASR-HF 原生支持要求 `transformers>=5.13.0`（由 `uv sync` 按 `pyproject.toml` 安装）。`pyproject.toml` 同时把 `torch` 钉到 `pytorch-cu130` 索引，请勿改用 `pip install` 逐个装依赖——那样会从 PyPI 取到 CPU 版 torch。

启动网页控制台：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run --no-sync python launcher.py
```

默认地址为 `http://127.0.0.1:2233`（SSE 用 2234）。端口被占用时会自动往后找下一个可用端口，实际地址在启动时打印；也可以用 `JAV_TRANS_PORT` / `JAV_TRANS_EVENTS_PORT` 指定。首次运行可以没有 `.env`；打开页面后在“翻译设置”面板填写 API Key、Base URL、模型和目标语言，保存或开始任务时会自动写入项目根目录 `.env`。目标语言选“简体中文”或“繁體中文”时，译文会统一转换到所选字形（模型偶尔会答成另一种），选“English”则原样保留。新建的 `.env` 只启用实际保存的本机值，ASR batch、显存预算等运行参数会以注释示例形式写入。国内网络下载 Hugging Face 模型较慢时，可在“识别设置”里填写代理协议、地址和端口。

**翻译后端支持**：通过 `TRANSLATION_BACKEND` 选择两种后端。`openai`——OpenAI 兼容 API（支持 Chat 与 Responses）。`llamacpp`——**本地翻译**：程序托管官方 llama-server 运行唯一内置的 [Hy-MT2-7B Q4_K_M GGUF](https://huggingface.co/tencent/Hy-MT2-7B-GGUF/blob/main/Hy-MT2-7B-Q4_K_M.gguf)（约 4.6GB，首次使用自动下载，不需要选型），需先 `winget install -e --id ggml.llamacpp`（Vulkan 构建，装完即在 PATH 上）或从 GitHub Releases 下载 CUDA 包并在设置里填路径（N 卡上更快）；8GB 显卡默认开 2 个本地并发槽。本地后端逐句翻译，不使用术语表、角色参考和全片上下文（这些只在 API 后端生效）；翻译开始时临时释放 ASR 显存、切回 ASR 时自动重载。详细配置和扩展指南见 [翻译后端架构文档](docs/translation-backend-architecture.md)。

Web 提交是否使用 CUDA 取决于后端服务进程是否能看到 GPU，而不是浏览器本身。完整 ASR smoke 应确认日志中出现 `cuda_available=True`、`device=cuda:0` 或 `actual_device=cuda`。
Web 会在模型要求检查中提示驱动过旧或 CUDA 初始化失败。

---

## 使用流程

1. 打开网页控制台。
2. 选择视频文件。
3. 选择字幕模式和翻译设置。
4. 选中的视频会立即进入右侧“待开始”列表；确认后点击“开始任务”。
5. 在输出目录查看 SRT、质量报告和日志。
6. 开了“生成质量报告（.md）”的任务，完成后任务卡上会多一个“📊 质检”按钮：点开就是分组后的质量报告，触发阈值的指标会高亮并在 tooltip 里给出阈值，另有断点类型分布、复读标记的两层对照和带时间码的问题样例表，可直接照着时间码到播放器里核对。没开开关的任务不会显示这个按钮。

从右侧任务列表删除已结束任务时，会清理任务临时目录；跨任务的 ASR 结果缓存（`tmp/asr_cache/`）按设计保留，不随任务删除。运行中的任务第一次删除只执行取消，进入“已取消”后再次删除才会清理临时目录。

勾选“不翻译（仅日文字幕）”时，流水线仍会执行切分、ASR 和字幕时间轴生成，但跳过 LLM 翻译，最终输出 `<视频名>.ja.srt`。这是验证本地切分 / ASR / 字幕时间轴链路的推荐 smoke 模式。

---

## 完整工作流

```text
视频输入
  -> 任务上下文 / 配置解析
  -> 音频抽取与标准化
     - async resample 保留源视频 PTS / edit-list 间隙，WAV 样本位置始终对应视频时间
     - 时间轴滤镜属于音频缓存键；规则变化会自动失效旧 WAV 与下游结果
  -> 切分（asr.chunking.plan_chunk_cuts）
     - ASR encoder 前向 -> CTC 对齐头 -> 每帧 blank 后验
     - 连续 blank 游程即停顿，切点落在停顿中央
     - 取 ASR_CHUNK_MAX_S 之内最靠后的停顿，块长跑满上限；上限即 encoder 音频窗口
     - 输出精确铺满 [0, 总时长]，相邻块共边，不丢任何音频
     - 每个切点记录来源（停顿中点 / 无停顿时的定长兜底），进质量报告
     - 未配置 ASR_ALIGNMENT_HEAD_PATH 时退化为定长切分
  -> ASR wav chunk export
  -> Qwen ASR text transcription
     - 每块的解码预算由自身时长派生（时长 × ASR_DECODE_TOKENS_PER_SECOND）
     - 音频装不下更多 token，所以这个界不会截断真实对白；用完预算即判解码失控
  -> CTC 强制对齐（asr.alignment）
     - 复用同一个 encoder 再跑一次前向（RTF 0.00069，约为一次解码的 0.56%）
     - 逐字时间戳 + 对齐分数
     - 起点/终点按对齐头自己判为 blank 的帧向外走，修正 CTC 尖峰造成的跨度内缩
     - 外扩不越过本块自身的音频；相邻块共边，越界会直接变成下一条 cue 的重叠
  -> 后置闸（asr.postgate）
     - 失控重复 / 语速不可能 / 与邻块重复 等只打标，不删除
     - 标记随 chunk 下发到 segment，只落在产物里，当前不参与任何删除决策
  -> Subtitle Layout v3_1
     - 20 个日文源字符与 7s 词语跨度联合软约束（一次 DP，不是先后硬切）
     - 候选仅来自句末/分句标点、>=0.6s 强停顿、0.12-0.6s 实测词间隙
     - 候选按证据强弱计分：句末标点 < 强停顿 < 分句标点 < 词间隙
     - 词间隙内部再按实测静音长度连续加权，宽的优先，不看谁更能填满行
     - 每条严格从首个实测发音字开始，声学终点固定在最后一个实测发音字
     - 显示终点可在其后的空白静音里最多多停 0.5s，且在下一条前 2 帧停住
     - 没有安全点就保留超限 cue；不按比例造切点、不截断到 7s、不向 blank 中点延展
  -> 可选 LLM 翻译
  -> SRT / bilingual JSON / quality report / logs
```

关键约束：

- **ASR 之前不做丢弃式判断。** 切分只决定边界，输出铺满整个文件。
- **音频切块与字幕切分是两层不同的切点，选法也不同。** 音频块要的是喂给 encoder 的最大上下文，所以取 30s 窗口内**最靠后**的合法停顿；八片实测改取最宽停顿会把块长中位数压到 18.4–22.4s，正落在 2026-08-02 实测伤转写的区间。字幕 cue 的切点不受此约束，它在候选里选**证据最强**的那个（见上文 Layout v3_1）。停顿的 blank 后验两层都没参与：`blank_runs` 只读 argmax，游程本身不带概率。
- 内部 cut 是一个共享绝对时间戳，不允许左右 chunk 各自修边。
- 20 字和 7 秒都是字幕 layout 的**软目标**，不是 ASR chunk 上限；实测时间轴是硬约束。
- Runtime 不使用具体词黑名单或时长启发式删除短促人声。
- 有完整实测映射时，字幕行严格取本行首个发音字的起点与最后一个发音字的终点；零宽标点留在文字里，但不偷取发音字符的时间。
- CTC blank / 词间静音是**可安全断句的证据**，不是下一条字幕的显示起点：音频 chunk 仍可切在 blank 中点，字幕上一条按前词结束、下一条严格从后词的实测起点出现，中间长 blank 保持无字幕。
- 成功的 `ctc_forced_alignment` 词时间会与原文字符位置完整绑定。映射不完整、只有一个不可再分的实测 token，或只有 `synthetic_proportional` 时间时，保留上游整段并显式标记；Subtitle Layout 不再使用比例退化。诊断工具产生的 `grok_stt_word` 遵守相同 measured-timestamp 合同。
- 后置闸的 `min_alignment_score` 保持关闭，因为该阈值尚未标定。
- 对齐头**默认启用**：本项目默认指向 HF 上的 JAV 非语义人声变体 `ctc_aligner_jav_vocalisation_v2.pt`。置空会回退到定长切分与上游粗时间轴；checkpoint 缺失、下载失败或损坏都只 warn 一次并降级，不会让转写失败。
- allocated/reserved/shared VRAM 只写运行诊断，不参与功能判断；显式 CUDA 请求不可用时直接报错，不回退 CPU。

---

## 模型架构

ASR 只有一个 repo：`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`，不提供更小的低显存档。

除 ASR 外**只有一个可训练组件**：CTC 对齐头。它是 ASR encoder 上的一个薄头，encoder 全程冻结。

```text
ASR encoder 输出 2048 维 @13fps（76.9 ms/帧）
  -> LayerNorm
  -> Conv1d(2048->512, 上采样 x2)   # 38.46 ms/帧
  -> 若干层小 encoder
  -> Linear(-> 字表 + blank)
```

- **上采样 x2 不是可选项**：CTC 每个输出 token 至少要一帧，而日语语速在 13fps 下每 mora 仅 1.6~2.2 帧。
- **CTC 目标是「字」不是 kana**，因此不需要 g2p 依赖。
- **当前项目默认头**是 `ctc_aligner_jav_vocalisation_v2.pt`：混合 Galgame、anime SFW 与 anime NSFW/JAV 域，纯声学词表不含标点类；训练目标会剥离纯非语义人声片段，并加入 blank-only 样本，因此呻吟帧被训练成 blank。ASR encoder 始终冻结。标点仍由 ASR 文本保留，并作为零宽字符穿过强制对齐与字幕层。
- HF 同仓库继续保留原 `ctc_aligner.pt`：它主要由 Galgame 字符 CTC + Grok 稀疏帧监督训练，带标点词表，适合希望保持原训练域和原输出刻度的用户。JAV 变体是并列工件，不会覆盖或改义这个原文件。
- **同一份输出有两个读法**：与文本对齐得到时间轴，blank 游程用来选切点。两个读法都不丢音频。

`forced_align` 在 `src/asr/alignment.py` 内自己实现，不依赖 `torchaudio`（本项目 Python 3.14，torch 所在索引上没有匹配的 torchaudio wheel）。正确性由穷举所有合法 CTC 路径的参照实现验证（`tests/asr/test_asr_alignment_head.py`）。

对齐头跟 ASR 权重一起发在 HF 上，不进本仓库：它是 encoder 专属的，换 encoder 即作废。`ASR_ALIGNMENT_HEAD_PATH` 接受两种写法：

```text
hf:<repo>@<commit sha>#<文件名>   # 默认；首次运行下载进 HF 缓存，之后离线
./path/to/ctc_aligner_jav_vocalisation_v2.pt  # 本地文件，覆盖默认值
```

默认值**钉死 commit sha 而不是 `main`**，换头是显式改配置的动作。

首次运行把默认变体下载到 **`models/ctc_aligner_jav_vocalisation_v2.pt`**（约 15.2MB，与 ASR 权重同目录，走项目代理设置），之后完全离线；对应的 `.revision` 文件记录 commit sha。checkpoint 的字节内容参与 ASR finalize 缓存签名，换头会自动让旧的对齐结果失效。

### 两个 CTC 头怎么选

| 用途 | JAV 非语义人声变体（本项目默认） | 原通用头（保留） |
|---|---|---|
| HF 文件 | `ctc_aligner_jav_vocalisation_v2.pt` | `ctc_aligner.pt` |
| 训练域 | Galgame + anime SFW + anime NSFW/JAV，约 151.5h | Galgame，含 Grok 稀疏帧监督 |
| 词表 | 纯声学字符，标点不占输出类 | 带标点字符 |
| 非语义人声 | 剥离目标 + blank-only 样本，专门压制呻吟转写 | 没有这轮 JAV 专项目标 |
| 推荐场景 | JAV、呻吟密集、需要稳定 blank 停顿结构 | Galgame/anime 通用对齐、复现旧结果 |

八片真实影片的同头验收中，JAV 变体相对原头的 blank AUC 为 `0.9609 vs 0.9384`，同等约 5% 误伤下的非语义人声召回为 `90.5% vs 72.9%`；起终点中位误差维持同量级。残余风险是它把 86 条词义候选判成 `blank=1.000`，其中 6 条被独立 Grok 时间轴证实为误伤，其余是无人反驳而不是已洗清。因此默认选择针对本项目成立，不代表它应替换所有场景的原通用头。

源码运行时可在 `.env` 显式切换：

```env
# 本项目默认：JAV / 非语义人声优化变体
ASR_ALIGNMENT_HEAD_PATH=hf:jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf@5a6a789ceb2f22d2b8606743b13a8159af218362#ctc_aligner_jav_vocalisation_v2.pt

# 回到原通用头（原发布 commit 保持不变）
# ASR_ALIGNMENT_HEAD_PATH=hf:jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf@68baee74dbed3bf98ba0545988278da8cff0e713#ctc_aligner.pt

# 本地 checkpoint 也可以
# ASR_ALIGNMENT_HEAD_PATH=./models/ctc_aligner_jav_vocalisation_v2.pt
```

字幕拆分默认同时使用 `20` 字与 `7.0` 秒软目标。两者只改变如何在已有实测边界中选点；调小会增加 cue 数，调大则减少 cue 数，都不会允许比例时间：

```env
SUBTITLE_MAX_SOURCE_CHARS=20
SUBTITLE_MAX_DISPLAY_DURATION_S=7.0
```

例如第 20 字附近没有安全点而第 23 字后有实测停顿，允许保留 23 字再切；整段没有安全点则整段不切。`SRT_LINE_MAX_CHARS` 只是单条字幕内部的换行宽度，和上述 cue 拆分目标不是同一个参数。

**同样合法的两个切点之间，选的是证据更强的那个，不是更能填满行的那个。** 优先级是句末标点 > ≥0.6s 强停顿 > 分句标点 > 普通词间隙；词间隙内部再按实测静音长度连续加权（0.12s 罚 0.36、0.60s 罚 0.20），因为词间隙唯一的依据就是那段静音，而 0.12s 的下限已经贴近连续语流里音节之间的间隔。八片实测：落在 0.12–0.2s 边缘静音上的切点从 399 降到 301（−24.6%），cue 总数 7,016 不变、超 20 字仍是 41 条、超 7s 仍是 34 条。把同样的加权用到所有类别的版本已被测掉：它会把 134 个切点从写出来的逗号挪到声学停顿上，等于拿语法换静音。

`SUBTITLE_LAYOUT_ENGINE` 与 `SUBTITLE_TIMING_MODEL` **不是开关，是产物上的版本戳**：没有任何代码按它们分派，值只会写进每条 cue 的 `layout_engine` / `timing_model` 字段，供审计和 A/B 认出「这批字幕由哪一版布局生成」。本项目只有一套布局，所以写入未知值会**直接报错而不是照单标注**——用它做回滚是做不到的事，静默接受只会得到一份谎报自己出身的产物。当前值是 `measured_safe_boundary_dp_v3_1`；上一版 `measured_safe_boundary_dp_v3` 同样会被拒绝，因为两者的切点落点有约 1.4% 不同，让旧名字通过等于把新布局的产物标成旧布局的。

### CTC 影子评测

新对齐头通过人工 A/B 前可以旁路运行：正式头仍生成字幕，影子头只复用同一份 encoder 特征计算边界差异，不会改写任何输出时间轴。配置示例：

```env
ASR_ALIGNMENT_SHADOW_HEAD_PATH=./models/ctc_aligner.shadow.pt
ASR_ALIGNMENT_SHADOW_ROOT=./tmp/cache/alignment_shadow
ASR_ALIGNMENT_SHADOW_MIN_DELTA_MS=20
```

正常运行真实 JAV 任务后，每个任务的结构化观察记录会持久化到 `tmp/cache/alignment_shadow/`。只有达到最小边界差值的片段才进入候选池；可按起点/终点分层生成等长、盲化的 A/B 审计页：

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run python tools\audits\generate_ctc_alignment_shadow_audit.py `
  --observations tmp\cache\alignment_shadow `
  --output-dir agents\temp\<timestamp>_ctc-alignment-shadow-audit\audit `
  --per-boundary 25
```

`ASR_ALIGNMENT_SHADOW_HEAD_PATH` 置空即关闭。影子 checkpoint 的字节内容参与 finalize 缓存签名；影子加载或比较失败只会留下状态记录，不会使正式转写失败。

---

## 默认配置

默认配置内置在 `src/core/config.py`，首次保存 Web 设置时会自动生成 `.env`。`.env` 只用于本机私密值和显式覆盖，不复制默认配置。通常只需要在 Web “翻译 API”面板填写：

- `API_KEY`
- `OPENAI_COMPATIBILITY_BASE_URL`
- `LLM_MODEL_NAME`
- 代理协议 / 地址 / 端口（可选，用于模型下载和 HTTP 请求）

“思考强度”是 `none` / `low` / `high` 三档，默认 `low`，它们是直接发给供应商的线上值。翻译只有一条路径：**首轮按所选强度翻完全片 → 本地检查源文回显、残留日文假名、术语表未生效和长度异常 → 只把标记出来的 id 集中复译**；大回复格式失败会自动拆小。复译后仍逐字回显日文源文会让任务失败，不会把它缓存成中文成品。

**复译用哪一档：跟首轮同档，但不低于 `low`。** 只有 `none` 会升档（`none`→`low`），因为升档的存在理由只对它成立——不思考首轮实测有 10.1% 的 cue 原样回显日文，而复译末尾那道闸门正是为此会让任务失败。`low` 首轮曾经一律升到 `high`，实测不划算：那一次请求花掉 22,585 思维链 token，占整片账单 25%，而首轮 11 次请求加起来才两倍于它；改成同档 `low` 只花 7,673，照样把 146 条标记全部修好，源文回显、假名残留、术语合规三项完全一致（0 / 0 / 100%），整片 ¥0.886 → ¥0.752、墙钟 315s → 170s。代价是成品里长度比例离群项 6 条变 12 条（诊断项，不是错译）。想买回旧行为就设 `TRANSLATION_REPAIR_REASONING_EFFORT=high`。

四个检测器都是纯文本的本地检查（不额外花一次模型调用去判断该不该花钱），但证明力不同：源文回显和术语表漏用是确定错误，假名残留与长度异常只是相关信号。**术语表检查决定了首轮档位能压到多低**：实测 `low` 会把 37 条术语 cue 里的 6 条译成术语表之外的词，而这种替换既不是回显、也不含假名、长度还一样——没有这个检测器，便宜档省下的钱正好是从“用户唯一逐字指定过的那部分译文”里扣的。

**“术语表”填的词一定压过程序自己挖的词。** API 后端每片会先花一次请求从整片源文里挖 15 个高频词当补充提示（这次请求同时承担 prompt 缓存预热，所以全片源文一部片只按未命中价买一次），而它挖出来的往往就是这部片自己的说法——你写 `ちんぽ-肉棒`，它可能挖出 `ちんぽ-鸡巴` 以及 `おちんぽ` / `ち○ぽ` 这些同词异写。凡是和你的术语表**指向不同译词**的条目都会被丢掉，打码字符（`○` `〇` `●`）按通配符比对，所以变体写法也挡得住；指向相同的变体保留。术语表留空时程序不会凭空产生用词意见。

**这一档直接决定单片价格。** 实测 1,396 cue 的一部片：输出 token 占总量 18% 却占约 91% 的成本，而输出里约 93% 是思维链。所以省钱只有三个杠杆——调低这一档、减少请求数（见下面的并发说明）、把任务排到 DeepSeek 的空闲时段（北京时间 9:00–12:00 与 14:00–18:00 之外，价格减半）。输入侧的全片 JSON prefix 缓存命中率约 96%，只占约 4% 成本，不值得优化。

`none` 是真正关闭思考（Chat 走 `thinking.type=disabled`，Responses 走 `effort=none`），最便宜，但单独使用时实测约 10% 的 cue 会原样回显日文——上面的自动复译正是为闭合这一点而存在。`minimal` 不是关闭值（在 OpenAI/Gemini/DeepSeek 上都是最小的**非零**思考预算），程序会拒收。旧值 `medium` / `max` / `xhigh` 一律读作 `high`：`medium` 从来不是 DeepSeek 接受的值，被静默忽略后请求实际就跑在 `high`。

注意**思考开启时 `temperature` 与 `top_p` 不生效**（DeepSeek 文档明确：接受但被忽略），只有 `none` 档它们才真正起作用。

ASR 显存自适应默认值已经内置。batch 或显存预算可通过“参数调优”里的环境变量覆盖，或手动编辑首次保存后生成的 `.env`。数值类 ASR 覆盖项写 `KEY=`（等号后留空）表示**用回默认值**；路径类覆盖项写空值则表示**清空该路径并关闭对应能力**，例如 `ASR_ALIGNMENT_HEAD_PATH=` 会关闭 CTC 对齐头。

默认配置：

```env
ASR_BACKEND=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf
ASR_BATCH_SIZE=auto
ASR_BATCH_SIZE_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=4
ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto
ASR_STAGE_WORKER_VRAM_RATIO=0.95
ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO=jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf=6144
ASR_STAGE_WORKER_RAM_RATIO=0.95
ASR_STAGE_WORKER_HEARTBEAT_S=10
ASR_STAGE_WORKER_OOM_RETRY_LIMIT=6
GPU_BATCH_PROFILE_ENABLED=1
GPU_BATCH_PROFILE_GROWTH_THRESHOLD=0.80
GPU_BATCH_PROFILE_MAX_ENTRIES=16
ASR_CHUNK_MAX_S=30.0
ASR_CHUNK_MIN_S=2.0
ASR_CHUNK_MIN_PAUSE_S=0.6
ASR_MAX_NEW_TOKENS=
ASR_DECODE_TOKENS_PER_SECOND=10.0
```

解码预算按每块自身时长派生：`时长 × ASR_DECODE_TOKENS_PER_SECOND`。日语快速语速约 10 音拍/秒，该 checkpoint 约 1 token/音拍（实测中位 1.00 字符/token），所以这是「这段音频最多能装多少 token」的上界，不会截断真实对白。`ASR_MAX_NEW_TOKENS` 留空即按时长派生；填数字则是硬上限，能压解码成本但会截断，**不要把它当默认值**——固定 128 在 30 秒块上等于 4.27 token/秒，而实测最快的块是 4.45。用完自己的预算说明模型吐得比人说话还快，计入 `decode_cap_truncations`（判为解码失控，不是「上限太低」）。

ASR stage 固定由统一 GPU worker 持有 CUDA：切分用的 encoder 前向、ASR 解码和 CTC 对齐都在同一个 GPU owner 进程里顺序执行，Web / 调度主进程只做任务编排、缓存索引和输出写入。OOM、CUDA 状态异常或超过 `ASR_STAGE_WORKER_VRAM_BUDGET_MB` 时会杀掉 worker，不会把 Web 主进程一起带崩。

`ASR_STAGE_WORKER_VRAM_BUDGET_MB=auto` 按物理 dedicated VRAM × `0.95` 计算软 OOM 线；**推荐 8GB 及以上**：1.7B 权重约 3.4GB 常驻，默认配置（batch 8 / 20 秒块）在 8GB 卡上峰值约 `5692MiB`。

`ASR_MIN_PHYSICAL_VRAM_MB_BY_REPO` 的 `6144MiB` 是**硬下限**而不是推荐值：低于它直接拒绝启动，等于它则需要 auto batch 降到 4~6 才能跑，余量很小。检查在模型加载前完成；shared VRAM 不计入可用预算，任何正的基线增量都立即视为 soft OOM，显式放大的 worker budget 和 CPU fallback 都不能绕过。监控不可用会直接停止。物理 RAM 使用按 `total-available` 计算，超过 `total × ASR_STAGE_WORKER_RAM_RATIO`（默认 `0.95`）同样停止。

GPU worker 默认每 10 秒输出一次当前阶段、总耗时和静默时长心跳。字幕 cue plan 会单独记录 timeline normalize、measured-safe-boundary DP、polish 和 finalize 进度。

对齐后的 segment 会按内容签名缓存。签名包含 ASR backend、字幕选项与参与结果的运行配置；只改输出路径一类不影响内容的设置不会让缓存失效。

每个 ASR chunk 的转写结果另有一层跨任务的内容寻址缓存（`tmp/asr_cache/<模型签名>/<音频sha256>.json`）：键为 chunk 音频 PCM 内容加模型与解码参数，与路径、chunk 序号和任务无关。同一部片重跑、崩溃续跑、甚至不同任务里字节相同的 chunk 都直接命中，整段跳过 encoder+decoder。超时与隔离结果不会入缓存；`ASR_RESULT_CACHE_ENABLED=0` 可关闭，`ASR_RESULT_CACHE_ROOT` 改位置。

`ASR_BATCH_SIZE=auto` 以 5600MB 下的 repo 默认表为基线，按显存预算比例放缩初始 batch。ASR text batch 发生 GPU OOM 时会重启 worker、降低 batch 并从结果缓存续跑。切分阶段按 `ASR_FEATURE_BATCH_SIZE`（默认 4）编码固定 30 秒窗口，OOM 时不参与自动降 batch，会直接停止而不是假装重试。RAM OOM 同样直接停止，不伪装成可由 GPU batch 修复的问题。

auto batch 会在 `tmp/cache/gpu_batch_profiles.json` 按 GPU、显存预算、模型、精度、attention 实现、chunk 时长和解码预算跨任务学习。profile 记录已验证安全 batch 与 OOM 不安全上界：阶段 peak allocated 低于预算 `80%` 时，在两者之间二分探测；尚无 OOM 上界时则向当前阶段上限折半推进，OOM 后本次任务仍先减半恢复，同时把安全值压到不安全值以下。chunk 时长是 profile 身份的一部分，因此不同显存和不同 chunk 几何各自学各自的上限，不需要手动配置。当前只覆盖 ASR chunk batch；显式数字 batch 不参与 profile 学习。

推理只需要 ASR Hugging Face 模型本身；同一份权重会被加载两处，切分阶段短暂加载一次取 encoder 特征（算完即卸，不与解码模型同驻），解码阶段再加载一次并一直用到对齐 pass 结束。**权重按需加载**：需要它的阶段自己加载，所以整片命中缓存的续跑完全不会加载模型。源码运行时如果本地没有模型，会按需下载到 `models/`。JAV 对齐头（约 15.2MB）默认从同一个 HF repo 按固定 commit sha 取，落在 `models/ctc_aligner_jav_vocalisation_v2.pt`；置空则**不报错**，切分退化为定长，Subtitle Layout 保留上游粗时间窗口且不再按比例新增切点。

阶段耗时表的各行严格加总为总计。ASR 在独立的 GPU owner 进程里运行，进程启动、环境传递与结果回传单列为「ASR Worker 启动与传输」；未归入任何具名阶段的剩余时间落在「其他」。

训练产物（CUDA feature cache、synthetic WAV、sequence JSONL、tensor cache、`datasets/train/...`）都不是运行依赖，不随源码分发。

---

## 字幕与文本策略

- ASR 文本会做 Unicode NFKC、空白归一、换行折叠和展示安全处理。
- Qwen3-ASR runtime 始终使用 Transformers 官方 `apply_transcription_request(audio=..., language=...)` 路径，不提供演员名 / 人名 context 提示分支。
- 字幕时间轴来自 CTC 强制对齐的逐字时间戳。对齐头未配置、或某段的实测映射不完整时，该段**整段保留上游粗时间窗口并显式标记**，不再按字数比例摊开。
- **整条都是非语义人声、且连续出现的 cue 会被丢弃**（默认开启）。ASR 会把呻吟转写成假名，强制对齐无法拒绝已经给定的文本，所以只能在成句之后按文本过滤。判定是「拆解」而不是「字符集」：整条 cue 必须能被无词义假名加一份显式的拟声词表完全消耗，剩下任何一个字就保留，因此 `ちんぽ`、`イッちゃう` 这类与呻吟共用假名的词不会被误删；`うん` / `はい` / `ふふ` 等应答与笑声另有白名单。**只删连续的**：孤立一条夹在对白中间更可能是真实反应，词表无法分辨，所以用上下文代替词表。实测一部真实影片 1983 条中命中 349 条、删掉 224 条（11.3%），其余 125 条因孤立而保留。`SUBTITLE_DROP_VOCALISATION_ONLY_CUES=0` 关闭，`SUBTITLE_VOCALISATION_MIN_RUN` 改连续条数阈值（设 1 即命中就删）。
- LLM 翻译前会先固定 cue plan，翻译不会重排时间轴。
- 最终中文输出遵循 Netflix Chinese (Simplified) TTSG 的**文本类**规则：每行 ≤16 全角单位、最多 2 行（下宽金字塔）、不用逗号句号（句中停顿为单个空格）、省略号为单个 U+2026、全角？！且不连用、半角数字、无斜体。标点归一化与折行在 `src/subtitles/zh_style.py` 的写盘层完成，翻译缓存保留 LLM 原文；质量报告以 `spec_*` 指标核查全部硬指标。`SRT_LINE_MAX_CHARS` 默认 16。
- **出点 +0.5s 这条已经恢复；最短 5/6s 与 2 帧最小间隔仍是被明知放弃的。** 三条规则被放弃的原因是它们都要求把 cue 边界移到没有语音证据的位置，但**出点是例外**：cue 之后的静音本来就空着（八片实测每片自由静音中位数 0.57–2.13s，只有 10.4% 的 cue 后面完全没空隙），多停一会儿不需要发明任何时间戳。所以显示终点现在可以延伸，上限四条同时生效——`SUBTITLE_LINGER_S`（0.5）、下一条起点前 2 帧、`SUBTITLE_MAX_DISPLAY_SHIFT_FROM_ACOUSTIC_END_S`（0.5，同时保证重复运行不叠加）、以及不越过 `SUBTITLE_MAX_DISPLAY_DURATION_S`（7s）。**起点、声学边界和任何词时间都不动**，加了多少记在 `display_shift_end_s` 里；下一条在 2 帧内开始时整条不动，绝不回缩去截断实测语音。八片实测：短于 5/6s 的 cue **487 → 198**，日文源 CPS>7 **40.6% → 27.9%**，在屏总秒数 21,115 → 23,733s，而超 7s 仍是 34 条、重叠仍是 0 对、间隔小于 2 帧的 556 对不变、文本逐字不变。剩下的 198 条与 556 对仍然是明知的偏离：它们后面根本没有可用的静音。质量报告里这两项因此**按份额告警而不是按条数**：条数随片长走（同样 5–10% 的比例，短片 21 条、长片 97 条），拿一个能放过长片的条数阈值去看短片，等于放过四倍的回归。`QC_MAX_SPEC_DURATION_UNDER_SHARE` 与 `QC_MAX_SPEC_GAP_UNDER_SHARE` 默认都是 `0.15`，高于现默认头在八片上的最高单片份额（10.6% 与 9.7%）并留出波动余量；超过就说明 cue 形状变了，而不是布局在按设计工作。**换头需要重新标定这两个数**（保留的带标点头在同样八片上 gap 份额是 12.6–23.8%）。其余 `spec_*` 仍是零容忍条数阈值，它们才是回归信号。出点延伸可以用 `SUBTITLE_TIMING_POLISH_ENABLED=0` 整体关掉（那会退回「cue 在最后一个发音字就消失」）；最短时长与 2 帧间隔则没有单独开关，要它们只能整体退回 v2 布局。
- 质量报告还记录三组**只观测、不告警**的痕迹，它们在成品 SRT 里已经看不出来。**字幕切点**：`layout_break_type_counts`（每条 cue 由哪种证据结束）、`layout_word_gap_cut_count` / `layout_word_gap_cut_under_0p2s` / `layout_word_gap_median_s`。**续句标记**：`cue_continues_from_previous_count` / `cue_continues_into_next_count` / `cue_continues_from_previous_share`，以及 `vocalisation_runs_dropped` 与 `vocalisation_continuity_flags_cleared`。两者其实是同一件事——`continues_into_next` 就是「这条不是以句末标点结束的」，所以 break type 的分布正是续句数量的成因；而 `vocalisation_continuity_flags_cleared` 是**撤回**的声明数，脱离声明总数读不出意义。**音频切块**：`chunk_cut_policy` / `chunk_cut_at_pause_count` / `chunk_cut_max_fallback_count` / `chunk_cut_max_fallback_share` / `chunk_duration_*`，其中硬切份额（窗口内没有停顿、只能按 30s 切）八片实测 0.7%–53%，高的那几片是整片连续人声而不是切得差，因此不设阈值。改选点规则或换头之后，这三组数是唯一能跨运行对照的记录。**出点延伸**另记 `display_linger_applied_count` / `display_linger_total_s`：它是 `spec_duration_under_min_share` 的成因，某次运行份额跳高而这两个数是 0，说明这一步没跑，而不是 cue 形状变了。**后置闸标记**同时给两层——chunk 级 `postgate_chunks_reviewed` / `postgate_chunks_flagged` / `postgate_chunks_flagged_share` / `postgate_chunk_flag_counts`（检测器看到什么），cue 级 `postgate_flagged_cue_count` / `postgate_flagged_cue_share` / `postgate_cue_flag_counts`（有多少真的活到成品字幕里，Markdown 里另有一张对照表）。只有 cue 级那一列才构成行动理由；`repeated_unit` 在本域本来就有约 10% 的 chunk 命中，且多数是真实的重复语气词，所以两层都不设阈值。`postgate_alignment_score_checked` 为 0 表示未标定的对齐分检查没有运行——这一项如实报告，免得被读成「每条 cue 都有音频支持」。

---

## 输出与缓存

- `video/<视频名>/`：正式字幕、质量报告和人工质检报告。
- `models/`：Hugging Face 模型缓存。
- `tmp/jobs/<job_id>/`：Web / pipeline 单次任务临时目录；`JOB_TEMP_DIR` 默认是 `./tmp/jobs`。
- `tmp/chunks/`：ASR wav chunk 的一次性运行目录。
- `tmp/asr_cache/`：跨任务 ASR 结果缓存；按内容寻址，任务删除时保留。
- `tmp/cache/alignment_shadow/`：可选 CTC 影子头的跨任务边界分歧记录；任务删除时保留。
- `tmp/cache/torch/`：torch 运行缓存。（Hugging Face 下载缓存在 `models/hub`、`models/xet`，属于模型权重，删掉要重下。）
- `tmp/log/<job_id>/`：默认启用的本地诊断目录；包含 `.run.log` 和持久化 `.timings.json`。
- `datasets/`：本地训练、验证、测试数据归档，默认 ignored；不进入 GitHub 源码仓库。
- `agents/temp/`：研究脚本、smoke、临时日志和中间产物。
- `agents/audits/`：可长期复查的本地审计页，默认 ignored，不随 `git push` 发布。

本地审计页服务：

```powershell
.\tools\audits\serve_audits.ps1
```

审计导航会显示每个审计产物的生成时间，优先使用 summary 时间，其次使用目录名前缀，便于区分多轮审计页。审计服务支持音频 Range seek 和导航页删除 API。直接打开 HTML 可以浏览页面，但删除按钮不能真正移动本地审计目录。

成功运行后默认删除一次性 job 临时目录；保留可复用缓存，例如 `models/`、`tmp/cache/` 和 Web 状态。

---

## 常见问题

### 模型下载慢

在 Web “识别设置”里填写代理，例如：

```env
PROXY_PROTOCOL=http
PROXY_HOST=127.0.0.1
PROXY_PORT=7890
```

或提前把模型下载到 `models/` 对应目录。发布版安装器在首次运行时也会写同样的三个键，安装依赖和下载模型共用这一份代理。

这一份设置覆盖所有出网请求：安装 PyTorch、下载 ASR 权重与对齐头、下载 GGUF 翻译模型、调用远程 LLM API。本机回环（`127.0.0.1` / `localhost` / `::1`）自动豁免，所以填了代理不会影响本地 llama-server。唯一不经过代理的是 `llama-server.exe` 本身——它由 `winget install -e --id ggml.llamacpp` 或 GitHub release 手动安装。

### CUDA 没有被使用

确认日志中出现：

```text
actual_device=cuda
model_param_device=cuda:*
```

受限 sandbox、错误的 PyTorch wheel、驱动问题或从非 GPU 环境启动 Web 服务都会使 CUDA 启动检查失败；正式工作流不会转到 CPU 继续执行。

### 显存不足

默认配置已按 **8GB 级显存**目标设置（见上文「默认配置」），硬下限是 `6144MiB`。

先让 auto batch 自己收敛：`ASR_BATCH_SIZE=auto` 会按 GPU、显存预算、模型和 chunk 时长学习安全 batch，OOM 一次就记下上界并二分回退，profile 跨任务保存在 `tmp/cache/gpu_batch_profiles.json`。8GB 和 16GB 卡各自学各自的，不共用。

只有在 auto 反复 OOM 时才手动钉值——显式数字会关掉学习：

```env
ASR_BATCH_SIZE=2
```

只有 1.7B 一档，没有更小的模型可切换。降 batch 之后仍然 OOM 就只能关掉其他占用 GPU 的程序；本程序不会改用 CPU 继续推理。

### 翻译报「response was cut off」

每个翻译请求的输出上限**不是** `TRANSLATION_MAX_TOKENS`（384000，只是给 API 模型的天花板），而是按这一批的源字符数算出来的：`源字符数 × TRANSLATION_OUTPUT_CHAR_RATIO + 结构开销 + 本请求所需的思考额度`。`none` 档不思考，因此该项为 0；其余档位都加入额度。两者取小，所以调天花板不会有任何效果，要放宽只能调 `TRANSLATION_OUTPUT_CHAR_RATIO`（默认 1.5）。

**开思考的请求仍要给思考留预算**：`max_tokens` 在推理模型上先花在思考流上，而前两项只建模了看得见的回复。实测 deepseek-v4-flash，一个 8 条 cue 的小批要思考 2,058 字符、54 条批 20,231 字符，都比按字符算出来的预算高一到两个数量级。所以思考档位固定加入 `TRANSLATION_REASONING_TOKEN_ALLOWANCE`（默认 32000）。思考量主要由强度决定、不与源字符成比例，所以它是固定额度而不是第二个比例系数；`low` 与 `high` 共用这一份，因为实测两者需求重叠、谁更省并不稳定。

撞上这个上限时会自动按 `TRANSLATION_TRUNCATION_RETRY_FACTOR`（默认 2.0）加大预算重试一次，并在运行日志里留下 `output_truncated` 事件。重试后仍被截断说明模型在重复自己而不是预算太紧——这正是这个上限要挡的事，此时调大 ratio 只会让失控跑得更久。放宽额度不会放松跑飞保护：真正拦住 `嗯嗯嗯…` 的是按批内最长源行给每条译文加的 `maxLength`，与 token 预算无关。

### 翻译报「invalid batch translation id」

模型返回的条数对，但 id 不在这一批要的范围内，通常是整批编号偏移了。程序**不会**接受这种回复：一个偏移的 id 集合会把每条译文挂到相邻 cue 上，而且看不出来。

API 翻译的每批条数就是 `TRANSLATION_BATCH_SIZE`（默认 200），**与并发数无关**；并发只决定同时有几个批在飞，且不会超过批的总数。2026-08-24 之前批大小是按 `⌈总条数 ÷ 并发 ÷ 2⌉` 算的，那让并发变成了一个计费旋钮：同一部 1,396 cue 的片，4 并发是 8 个请求、16 并发是 32 个，思考账单差 4 倍却是同样的活。

**请求数是 API 成本的主要驱动量**：一次请求的思考量近似固定，不随批大小成比例增长（实测 24 条 cue 花 18,393 字符、54 条花 20,231），所以整片的思考账单跟着请求数走，而不是跟着 cue 数走。调并发现在只影响墙钟时间，不影响账单。Web 页面在并发输入框下方显示同一说明。

**批大小该按「一次能不能答完」来定，而不是按预算算出来。** 四次整片运行、每批 200 条，32 个请求里有 7 个没答完——总是丢掉末尾一段连续 id（分别缺 9、50、100、100、184 条），或者返回了不在本批范围里的 id。当时输出预算是 42,495 token、实际最多只用到 31,486，**所以不是被截断，是模型自己提前收尾**，条数越多越容易发生。这种失败在成本上很贵：废掉那次请求的思维链要照付，补发还要从头再想一遍。

格式失败会让下一次请求**减半**（批内只降不升），直到降到模型能抄对 id 的规模，日志里对应 `batch_span_narrowed` 事件；剩下的 id 走正常的补发路径。所以一般不需要手动干预。如果供应商反复无法完成大结构化回复，可以在 `.env` 里降低单请求安全上限：

```env
TRANSLATION_BATCH_SIZE=24
```

可配置范围 8–400。调低后每条回复要抄的 id 更少、一次答完的概率更高，但请求数上升，而每个请求都要付一份几乎固定的思维链——所以这是一个权衡，不是单方向的优化。

### 同一部片重跑，字幕为什么不完全一样

因为**一次 `generate` 里同批有哪几个块，会改变每个块的解码结果**。解码是贪心的（`do_sample=False`，模型自带的 `generation_config.json` 里没有任何采样参数，与官方推理封装写死的 `temperature=0.0` 等价），但同批的块要补零对齐到最长的那个，批大小一变，bf16 下的累加顺序跟着变，个别位置的 argmax 就会翻过去。

实测同一部片（151 分钟）关掉结果缓存重跑一次：音频切块的 339 个边界**逐条相同**，262 个块的转写文本不同，总字符差 1.0%、cue 数差 1.7%——两次运行的差别只有批大小（11 与 5）。把同样的前 30 个块重切出来单独复解可以两头对上：

- **批大小固定就逐位可复现。** batch=5 连解两遍，30 个块的文本完全相同；和几天前那次 batch=5 整片运行的存档逐条比对，也是 30/30 一字不差（不同进程、不同日期）。
- **批大小一换就变。** 同样这 30 个块改成 batch=11，20 个块的文本不同。

默认 `ASR_BATCH_SIZE=auto` 并不保证两次运行拿到同一个值：有匹配的显卡档案时直接用学到的值（实测 11），档案不匹配时退回按显存推算（实测 5）。要让重新解码可比，就在 `.env` 里钉死。**钉哪个数请照抄本机实际用过的值**——它写在 `tmp/log/<job_id>/<...>.timings.json` 的 `asr_details.stage_worker.runtime_tuning.asr_batch_size` 里：

```env
ASR_BATCH_SIZE=11
```

不要随手往小了钉。批越大吞吐越高（同一部片实测 batch=11 约 1.2s/块、batch=5 约 2.4s/块），钉一个比本机档案小的值等于白白慢一倍；钉一个比档案大的值则可能 OOM。

跨任务的 ASR 结果缓存（`tmp/asr_cache/`）仍然是更省事的办法：全部命中时根本不会再解码，重跑会拿到**一模一样**的转写，改翻译设置重试也不会连带换掉日文文本。要对比两种配置的字幕差异，请让它保持开启（默认开启）。注意**部分命中反而会改批的组成**——命中的块不进这一批，剩下的块就被重新分组，所以「缓存半满 + 没钉批大小」是最容易得到第三种结果的组合。

### 一部片大概要跑多久

整片耗时的大头通常仍是 ASR 解码。151 分钟样片在 RTX 4060 Ti 上冷跑 920 秒（约 9.9× 实时），其中文本转写 798s、静音切块 23s、字幕时间轴 32s、旧不思考翻译 47s、音频准备与写出各不到 6s。旧全量思考臂的翻译耗时为 `low` 238s / `medium` 456s / `max` 599s；这些数字不再代表当前生产路径。

当前默认（Responses、`low`、4 并发、复译同档）已在样片的 1,595 条 cue 上实测：DeepSeek 总翻译约 170s、11 次请求、65,764 输出 token（其中 33,125 是思维链），按官方价高峰约 ¥0.75 / 空闲约 ¥0.38，术语表合规 100%，成品零源文回显、零假名残留，本地质量门终态只剩 12 条长度诊断项。输入侧缓存未命中只有 25,255 token、首轮各批命中率 99.4%——因为术语提取请求与所有批共用同一段前缀，全片源文一部片只按 miss 买一次。耗时会随首轮可疑项数量显著变化：这一次首轮 359s、选择性复译 67s。相同 cue 的本地 Hy-MT2-7B Q4 在两个 llama.cpp slot 下用时 249.1s，但仍有 38 条残留日文、11 条源文回显，不能把更快理解成质量等价。

**别用单次运行去比 ±30% 的成本差。** 三次配置完全相同的运行，首轮思维链分别是 38,641 / 87,727 / 45,718 token（跨度 2.3 倍），单个批次从 15 token 到 28,555 token 都出现过——同一档位下模型想多久是它自己决定的。要比档位就多跑几次取量级。

### 长任务怎么排查

运行日志默认写入 `tmp/log/<job_id>/`。`.run.log` 便于查错，`.timings.json` 记录音频准备、切分、ASR、字幕时间轴、翻译、写出等阶段耗时和显存快照；Web 完成任务后也会把这两个文件列在“其他文件”里。反馈问题时请保留 `.run.log`、`.timings.json`、质量报告和对应 SRT。

---

## 开发

主要代码位置：

- `src/main.py`：主流程编排。
- `src/core/`：配置和任务上下文。
- `src/pipeline/`：音频、缓存、输出、质量报告和阶段日志。
- `src/asr/`：ASR 转写、切分（`chunking.py`）、cue 特征（`cue_features.py`）、CTC 对齐头（`alignment.py`）、字幕时间轴（`subtitle_timing.py`）与后置闸（`postgate.py`）。
- `src/llm/`：翻译侧三层——`backends/`（transport）、`profiles/`（各模型家族 prompt 合同）、`engine.py`（唯一编排循环），`translator.py` 是门面；另有 prompt、cache、glossary、修复批与术语预抽取。
- `src/subtitles/`：SRT writer、字幕选项和字幕 QC。
- `src/web/`：FastAPI 接口和静态前端。
- `tools/`：对齐头训练、审计页、离线 Teacher、ASR SFT 和 workflow smoke 工具。

常用测试：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run pytest tests/asr           # 转写、切分、对齐头、后置闸
uv run pytest tests/pipeline      # 编排、GPU worker、batch profile、产物写出
uv run pytest tests/llm           # 翻译 backend/profile/engine 与缓存
uv run pytest tests/subtitles tests/web tests/tools
uv run pytest                     # 全量
```

---

## 工具索引

所有 Python 工具都从项目根目录执行，并使用当前 `.venv`：

```powershell
$env:PYTHONIOENCODING='utf-8'
uv run python -m <module> --help
```

常用入口：

- `tools.workflows.run_full_workflow`：命令行完整工作流 smoke。
- `tools.workflows.promote_torch_checkpoint`：晋升生产 checkpoint。
- `tools.web.smoke.start_server` / `submit_job` / `poll_job` / `summarize_job`：Web 服务 smoke 和任务汇总。
- `tools.sft.*`：ASR 本体的自训链路（线上默认权重 `…-JA-Anime-Galgame-hf` 就是它的产物，不是退役件）。`export_qwen_asr_sft`（从物化的 HF 音频 manifest 导出可上传的音频/文本行）、`prepare_qwen_asr_sft_dataset`（Galgame ASR/SER 数据生成可复现的 SFT JSONL，`--mode smoke|full`）、`train_qwen_asr_sft_hf`（原生 Transformers 训练入口）、`probe_qwen_asr`（拿 manifest 直接跑本地权重做转写抽查）。
- `tools.align.*`：CTC 对齐头训练链。
  - 数据与训练：`build_alignment_features`（encoder 特征抽取）、`build_real_alignment_manifest` / `build_real_alignment_lines`（真实数据 manifest）、`run_grok_ctc_teacher` / `select_galgame_ctc_teacher_pilot` / `expand_galgame_ctc_teacher_pilot` / `frame_teacher_supervision`（Grok 词时间与稀疏帧监督；`compile_sparse_frame_targets` 的 `start_offset_s` 说明这批帧从 clip 的哪里开始，crop 行不给就会整体前移且不报错）、`train_ctc_aligner`（训练。帧教师按 **`--frame-teacher-cache`** 声明、可重复：归档覆盖的是一个缓存，而域是损失配比标签、会同时容纳有时间轴的行与本就没有时间轴的空白行；被声明的缓存必须整份被覆盖，否则直接停，未声明的缓存只带 CTC 文本损失并在 summary 里显式报数）。
  - 语料构造：`build_vocalisation_stripped_manifest`（把脚本按标点切块、丢掉纯非语义人声块，只留语义部分作 CTC 目标——呻吟帧因此只能被解释成 blank；剥离方向是安全的那一侧，分类器拆不开的一律保留）、`audit_stripped_fragments`（审剥离掉的是什么：带汉字的移除数应为 0，并用本地 ASR 独立读回作旁证）、`build_vocalisation_blank_manifest`（脚本本身就是纯人声的片子直接作 blank-only 行，**文本必须清空**，否则是在训练头去对齐呻吟）、`prepare_alignment_cache_manifests`（每个缓存一份 manifest，不跨运行合并——各运行的组编号独立，合并会让同名组跨 train/val）、`build_alignment_caches_v2.sh` / `train_ctc_aligner_v2.sh`（现役特征缓存与训练的复现脚本）。
  - 评估：`evaluate_alignment_geometry` / `evaluate_ctc_cache`（几何与冻结缓存）、`evaluate_pregate_loss` / `measure_pregate_dropped_audio`（切分与丢弃音频）、`build_ctc_ab_jav_predictions`（固定文本与声学窗口，只让两版头产生边界差，供盲化 A/B）。
  - 换头验收（一次编码、多头同测，所以比的是头不是数值环境）：`compare_heads_on_film`（真实片上按 cue 读自由跑 argmax 的 blank 占比，给出人声 vs 词义对白的 AUC、各阈值召回/误伤，并导出逐 cue 配对值——组中位数只说刻度动了，配对值才说某条是不是变坏了）、`aggregate_head_acceptance`（多片合并；AUC 在合并后的 cue 上重算而不是按片平均，同时保留逐片表，因为「合并赢」和「每片都赢」是两种结论）、`adjudicate_silent_cues`（头判 100% blank 的词义对白条，用 Grok 词级时间轴独立裁决——**本域不能默认「ASR 写了汉字就是真话」**，`気持ちいい` 是 JAV 上 ASR 最可能吐出的词；该片没有任何 Grok 词义词时直接报错，不报 0 条误伤）、`realign_segments_with_head`（冻结转写只换头重对齐，供时间轴对比；不复现生产的边缘外扩，故绝对值与生产不同，但两头同等处理）、`compare_head_to_teacher`（与 Grok 按语音岛比边界，只取互唯一重叠对）、`measure_pause_structure`（标点类会不会把 `blank_runs` 的停顿劈碎：同一份 argmax 读两遍，严格口径只认 blank、宽松口径把标点也当 blank，报「因碎片化而低于 `ASR_CHUNK_MIN_PAUSE_S`、切分器彻底看不见的停顿」；纯声学词表的头两口径恒等，正好当对照）。
  - 测量：`measure_blank_class_separation`（在干净 galgame 的 Grok 无词区按能量拆出 `voiced_wordless` 与 `silent`，与 `word` 一起给出闸门余量 `margin_vs_non_semantic_pp`；默认只跑 val，因为 train 的长空隙本身就是 blank 监督）、`measure_core_leading_silence`（clip **两端**自带的静音，用于把它从起止点误差里减掉；两端都向「语音更长」取整，所以「走到语音外面」的份额一律是上界，而「切掉语音」不会被它虚报）、`sweep_edge_caps`（**用实测语音起止点选 `ONSET_BACKOFF_MAX_S` 与 `CODA_EXTEND_MAX_S`**：一次前向、两端各自扫，因此各档之间没有采样噪声。起点找拐点，因为提前近乎免费；终点用双边夹逼，因为两个方向都要钱——下界看「结束落在语音内」，上界看 `share_past_core_end`，后者不需要检测器，因为超过 `core_end_s` 就已证明走出了 clip）、`sweep_blank_bias`（Viterbi blank bias 扫描，实测无可用工作点，默认 0.0）。
  - 真实域教师归档与准入：`archive_grok_fullfilm_teacher`（把整片 Grok STT 运行归档成训练数据源，保留付费响应与绝对词时间，不复制源视频）、`audit_teacher_silence_against_head`（**用作 blank 负样本前的准入闸**：拿生产头同一段音频的 `aligned_segments` 反查「教师沉默」，判据是**会吞掉生产头多少自有语音**而不是「争议占 blank 多少」。**「语音」按「说了什么」计，不按「吐了字符」计**：强制对齐无法拒绝 ASR 为呻吟写的假名，而教师是有意丢弃它的，所以纯非语义人声的语音岛在比较前先从头这一侧排除，`--count-vocalisation-as-speech` 可还原 v2 读法审计。**两侧文本从不互相比对**——不同转写器对同一段音频用词必然不同，跨教师比文本等于凭词汇制造分歧；只比时间，文本只在头这一侧读，且只用来判断「头是否声称这里说了话」。裁决三态且带 `scope`，没有「通过」这一态——`reject` 退出码 2、`no_conflict_observed` 0、`inconclusive` 3；所有统计量都按实际比较的秒数计算，前缀结论不能当整片结论引用）。
  - 已退役但保留可测：`pregate_reference`（被证伪的前置闸读法，留着继续被度量，不在转写路径上）。
- `tools.audits.audit_nav` / `serve_audits.ps1`：审计页导航与 Windows 本地服务（脚本实际拉起的是 `tools.audits.serve_static`，需要直接调用时用它）。
- `tools.audits.review_page_core` / `audit_prompt` / `binary_clip_audit`：人工审计页共享 Core（`MM:SS.mmm` 区间显示与播放器、状态、完成度、保存 API）与可复用提示配置；任务特有布局与 verdict 组合由 Adapter 提供。设计合同见 [Human Audit Page Core](docs/audits/20260723_human-audit-page-core-v1.md)。
- `tools.audits.select_alignment_onset_audit` / `generate_alignment_onset_audit_html` / `evaluate_alignment_onset_audit`：对齐头起止点人工审计的抽样、页面生成与统计。
- `tools.audits.generate_ctc_alignment_shadow_audit`：从正常真实 JAV 任务留下的影子分歧记录中，按起点/终点分层抽样并生成盲化 A/B 审计页。
- `tools.audits.generate_ctc_alignment_ab_audit` / `evaluate_ctc_alignment_ab_audit`：为两版 CTC 头生成真实音频盲化 A/B 页面并统计人工裁决；`generate_galgame_ctc_teacher_audit_html` 用于 Galgame 教师词时间审计。
- `tools.audits.generate_subtitle_ab_compare_audit_html`：两版字幕的 A/B 对照审计页。
- `tools.audits.generate_translation_ab_audit_html` / `evaluate_translation_ab_audit`：两套翻译配置的**逐 cue 盲化 A/B** 页面与统计。两臂就是同一部片的两次运行，**cue 数、每条起止点与日文原文必须逐条相同**，否则直接报错——最省事也最正确的造法是任务跑完后改「翻译设置」再点「重试翻译」，重试复用 ASR 产物，几何天然一致；重新解码过的运行不能当臂（同一份音频重解会有约 1% 字符差）。只抽两臂中文**确实不同**的 cue，甲/乙 顺序按半数平衡随机，答案只写 `answers.jsonl`、页面生成前按结构校验不含臂身份。统计只在**分出胜负**的卡片上做符号检验与 Wilson 区间，「都可用」不折半计入，未审阅数如实报出。
- `tools.subtitles.build_dual_track_ass`：把两份 SRT 叠成一个 ASS，旧的贴顶、新的贴底，供直接挂在片源上实听对比。**两条轨道刻意不配对**——各自保留原时间轴作为独立事件流，否则「谁在什么时候出现」这个最该看的差异会被配对逻辑抹平。同时报出各自的 cue 数、在屏秒数与字符数。
- `tools.audits.select_pause_frame_audit` / `pause_frame_audit` / `generate_pause_frame_audit_html` / `generate_pause_frame_review_html`：真实域 safe-cut 帧标注的抽样、标签合同与只读复核页。标注页刻意不显示任何模型输出（blank 游程摆在问题旁会制造一致性）。**该问题现已由 `tools.align.measure_blank_class_separation` 在干净 galgame 上自动回答，页面保留作为将来确需人耳裁决时的设施**。
- `tools.audits.grok_stt_smoke_audit`：小批量 Grok STT 词时间审计；OpenRouter 默认响应只有 `text`/`usage`，必须显式请求 `verbose_json` 与 `timestamp_granularities=["word"]` 才有词时间。
- `tools.datasets.label_drop_spans_words` / `apply_drop_span_relabels` / `cut_long_drop_span_clips`：drop-span 逐词 Teacher 标注、复核回写与长片段切分。
- `tools.audits.build_word_definition_calibration` / `evaluate_word_teacher_calibration`：逐词 Teacher 的「什么算词」校准集与一致性评估。
- `tools.omni.run_audio_teacher` / `audio_teacher_batch`：离线音频 Teacher Core；统一处理 `--prompt/--prompt-file`、`--folder/--file/--manifest`、provider-safe 并发、进度、续跑和主线程串行化落盘。
- `tools.omni.audio_teacher_transport`：Qwen、OpenRouter、Google AI Studio 三个 provider Adapter 的唯一分派入口；`--env-file qwen|openrouter|gemini` 只接受这三个已知 profile（`~/.config/omni/` 下的隔离配置），请求与响应协议互不冒充。
- `tools.omni.speech_to_text_transport` / `run_grok_stt_fullfilm` / `build_grok_stt_srt`：Grok STT Adapter、可续跑的整片分段转写与生产 Subtitle Layout SRT。整片工具默认 `x-ai/grok-stt-1.0`、diarization、5 分钟块 + 5 秒 overlap、$10 预检上限；分段音频用 async resample 保留视频 PTS 间隙，换人只在相邻发言不重叠时形成切点。
- `tools.omni.gemini_native` / `inspect_gemini_quota`：Google AI Studio 原生 Interactions 音频 Adapter；内联音频、结构化输出、多 Key 轮换与保守的本地滚动配额账本（多 Key 只增加配额轮换槽，不增加并发；`inspect_gemini_quota` 不发请求，只显示脱敏状态）。
- `tools.omni.timestamp_contract`：Teacher 时间坐标的严格 `MM:SS.mmm` wire schema、格式化、解析和 source-bound 校验；不提供数字秒兼容或时间猜测。
- `tools.sft.*`：Qwen3-ASR SFT 自训链路——数据集准备、云端训练资产与训练脚本；线上 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` 即该链路的发布产物。
- `tools.asr.convert_qwen3_asr_to_hf`：把 Qwen3-ASR 权重转换为 HF 布局；`tools.asr.measure_repetition_budget` 测量解码预算与重复率，用于标定重复守卫的阈值。
- `tools.omni.openai_compat`：内联音频的流式 chat-completions 共享层（env-file 加载、从散文里抽 JSON、ffmpeg 切片），由上面各 omni Teacher 复用。

命令行完整工作流 smoke：

```powershell
uv run python -m tools.workflows.run_full_workflow --video video/<your-video>.mp4 --task-name 20260617_191654_cli-smoke --label smoke
```

训练细节、实验记录和路线演变不在 README 展开；见 [docs/HISTORY.md](docs/HISTORY.md)。
