# 进阶设置与排错

[返回使用说明](../README.md) · [启动与下载](#启动与下载) · [API 翻译](#api-翻译) · [本地翻译](#本地翻译) · [显卡与取消](#显卡与取消) · [结果与恢复](#结果与恢复) · [反馈问题](#反馈问题)

本文补充 README 没有展开的设置、排错、任务恢复和源码运行方法。第一次使用请先按[快速开始](../README.md#快速开始)，用标准预设处理一个短视频。

## 启动与下载

### 程序窗口打不开

在程序目录打开 PowerShell，运行自检：

```powershell
.\jav-sub.exe --doctor
```

自检会逐项检查程序文件、FFmpeg 和依赖（包括实际导入 PyTorch 等库），缺少依赖时当场补装，然后继续启动。提示“程序文件不完整”时，缺的是压缩包里的文件，无法自动补回：把同一版本的压缩包按[升级步骤](../README.md#升级到新版本)重新复制一次即可。

其他启动参数：

| 参数 | 作用 |
| --- | --- |
| `--update` | 检查 GitHub Releases 最新稳定版并在线下载覆盖升级 |
| `--keep-console` | 程序窗口打开后保留命令行窗口，便于查看启动日志 |
| `--proxy <地址>` | 使用指定代理，不再询问 |
| `--yes` | 自检、升级和安装过程中不等待确认 |
| `--reinstall` | 重新安装运行环境 |
| `--install-only` | 只安装、自检或应用更新，不启动程序 |

这些参数针对发布包；从源码运行见[从源码运行](#从源码运行)。

发布包已包含 FFmpeg Shared。启动器优先使用本机已有的 uv，找不到时从 PyPI 下载到程序目录的 `bin/`，再准备 Python 和锁定版本的依赖。模型不在发布包内，首次运行对应任务时下载。

### 下载慢或连接失败

首次安装前，启动器会实测下载速度并询问是否使用代理。安装完成后，也可以在控制台的“网络代理”中选择 HTTP、HTTPS 或 SOCKS5，填写地址和端口并打开开关，再点“测试连接”检查能否通过该代理访问 HuggingFace。地址栏只填主机名或 IP，协议和端口分别填写；SOCKS 请选择 SOCKS5。启动时同样可以直接指定代理：

```powershell
.\jav-sub.exe --proxy http://127.0.0.1:7890
# SOCKS5 示例
.\jav-sub.exe --proxy socks5://127.0.0.1:1080
```

端口仅为示例，请换成本机代理实际监听的端口。HuggingFace 测试成功不代表 PyPI 或 API 服务也能访问；依赖安装、模型下载和翻译报错需要分别检查对应的服务。

SOCKS5 可用于首次安装、下载测速、模型下载和 API 请求。旧源码环境出现缺少 `socksio` 的提示时，在项目根目录运行 `uv sync` 后重启；发布包请按升级步骤更新。

`--proxy` 优先于已保存的代理，仅指定本次启动使用的地址。网页修改代理后，后续模型下载使用新配置，正在进行的下载保留原连接；关闭代理会清除已保存的地址和端口。“测试连接”会先保存配置，测试期间锁定代理输入项，修改配置后需要重新测试。

安装时命令行窗口会显示依赖下载进度，中断后重新启动即可继续。发布包本身只有约 100 MB，但装好运行环境和模型后至少需要 17 GB，请按后者预留磁盘空间。

### 在线升级与自动检测

发布版内置在线升级（源码 git 工作副本不执行覆盖升级，请继续使用 `git pull`）：

- **自动检测开关**：默认在启动时后台检查 GitHub `releases/latest` 的 `x.y.z` 正式稳定版（忽略草稿与预发布标签）。可在右上角版本号弹窗取消勾选「启动时自动检测新版本」，或在 `config/config.json` 中设置 `"JAV_SUB_DISABLE_UPDATE_CHECK": "1"` 关闭；关闭后仍可随时手动点击「检查更新」或运行 `.\jav-sub.exe --update`。
- **校验与覆盖流程**：下载会复用已配置的 HTTP/HTTPS/SOCKS5 代理，并校验压缩包 SHA-256 摘要与 `release-manifest.json` 清单；解压时会自动剥离外层 `jav-sub/` 目录。仅在无进行中任务、无重试恢复且同目录没有其他运行窗口时启动覆盖：升级期间会暂时暂停接收新任务，主窗口退出后由启动器按新版 `release-files.json` 覆盖程序文件、替换 `jav-sub.exe`（旧副本 `.old` 于重启后自动删除）并清空 `tmp/updates/`，保留 `.venv`、`.python/`、`models/`、`config/`、`outputs/` 与 `cache/`。若某文件因占用导致覆盖中断，下次重新启动会自动续装。

## API 翻译

程序使用 Responses 协议，服务必须提供 `/responses`；只支持 Chat Completions 的接口无法使用。

### 地址与模型要配套

| 服务 | Base URL 示例 | 模型名示例 |
| --- | --- | --- |
| OpenRouter（默认） | `https://openrouter.ai/api/v1` | `deepseek/deepseek-v4-flash` |
| DeepSeek 官方 | `https://api.deepseek.com` | `deepseek-v4-flash` |

表中仅为示例，可用模型以服务当前返回的列表为准。更换服务后，先保存 API Key 与 Base URL，再点“获取”重新选择模型并保存；不要沿用另一家服务的模型名。

`/responses` 返回 404 或 `Not Found` 时，依次检查 Base URL、模型路由，以及服务是否支持 Responses；仅凭一次 404 无法判断是哪一项出了问题。确认服务只支持 Chat Completions 时，需要更换服务。

### 结构化输出报错

程序默认要求译文按 JSON Schema 返回；DeepSeek 官方地址改用较宽松的 `json_object`。兼容服务不支持严格 schema 时，在程序目录的 `config/config.json` 中添加：

```json
{
  "LLM_STRUCTURED_OUTPUT": "json_object"
}
```

如果希望 OpenRouter 只把请求发给声明支持强制 schema 的供应商，设为 `"LLM_STRUCTURED_OUTPUT": "json_schema"`。这时若模型没有可用的 `structured_outputs` 路由，请求同样会返回 404。

### 翻译与上下文

API 翻译与本地翻译共用同一套精简 JSON 批翻译与局部复译流程（每条原文仅携带 `id`、`ja` 及可选的续句标记 `cont_prev`/`cont_next`，不含冗余时间字段），均支持用户术语表。区别在于上下文与批大小：API 默认 200 条/批并可使用全片前缀和角色参考；本地翻译默认 32 条/批（2400 字上限），仅携带窗口化邻句上下文（半径按单槽上下文自动取 4 或 6）。

翻译优先保留否定、条件、数量、施受关系和语气。同一句日语在不同语境中可以有不同译法；正常的短译文不会因为长度比例异常而被扩写。中文只在同一条字幕内换行，不会重新切分时间轴。

## 本地翻译

默认翻译后端为本地翻译（`TRANSLATION_BACKEND=llamacpp`，固定使用 `Hy-MT3-7B Q4 · llama.cpp`），网页上无需配置 `llama-server` 路径：

- **自动下载与查找顺序**：本地翻译启用且 `./models/runtime/llama-cpp/` 尚未就绪时，程序会在后台自动从 GitHub Releases 下载匹配当前 PyTorch CUDA 主版本的 Windows x64 CUDA 构建（约 226 MB，内置稳定兜底版本）。运行时按 `LLAMACPP_SERVER_PATH`（显式覆盖） → `./models/runtime/llama-cpp/`（受管目录，升级时随 `models/` 保留） → 系统 `PATH` → 首次任务兜底下载的顺序查找。
- **完整性与 CUDA 检查**：下载复用 `config/config.json` 中的代理配置并校验 SHA-256，随后使用 `.venv` 内 PyTorch 的 CUDA 运行库运行 `--list-devices` 检查 GPU 可用性；若未识别到 NVIDIA CUDA 设备会自动尝试补齐同版本 `cudart` 或回退到兜底版本，不会静默切到 CPU。如需关闭自动下载，可在 `config/config.json` 中设置 `"LLAMACPP_AUTO_DOWNLOAD": "0"`。

翻译模型固定为 `Hy-MT3-7B-Q4_K_M.gguf`，首次翻译时自动下载，约 4.8 GB。服务使用 `q8_0` KV 缓存并开启 `--kv-unified` 共享池，`LLAMACPP_CTX_SIZE` 与 `LLAMACPP_PARALLEL` 默认均为 `auto`：启动时根据 GGUF 模型头信息与当前可用显存自动选档（8 GB 显卡常用 `4096 × 2 槽`，显存紧张时自动降为 `4096 × 1 槽`，10–16+ GB 显卡自动扩展至 `6144/8192 × 3–4 槽` 并放宽邻句上下文；若实时显存不足以容纳最小档位会直接报错；也可在 `config/config.json` 填整数手动覆盖）。

本地翻译与 API 共用 JSON 结构化输出和局部复译检查（空译文、原文回显、假名残留、术语表未命中会自动重试）；可通过 `LLAMACPP_TRANSLATION_BATCH_SIZE`（默认 32）调整本地批大小。7B 模型对省略主语和复杂语境仍可能误判，重要字幕建议复查。

### 手动覆盖 llama-server

若因离线环境受限或需测试特定版本，可以从 [llama.cpp 官方 Releases](https://github.com/ggml-org/llama.cpp/releases) 手动下载 Windows x64 CUDA 版，直接解压到 `./models/runtime/llama-cpp/` 即可生效；也可在 `config/config.json` 中设置 `"LLAMACPP_SERVER_PATH": "<目录或 llama-server.exe 路径>"` 覆盖受管目录。

### 切换后端与等待关闭

任务运行中切换翻译后端，会停止受影响的任务并说明原因。保存新设置后重新运行，可复用仍然有效的缓存。

| 设置 | 默认值 | 作用 |
| --- | --- | --- |
| `LOCAL_BACKEND_WAIT_TIMEOUT_S` | 600 秒 | 任务等待可用本地实例的上限；超时后该任务报错并停止等待，旧实例继续清理 |
| `LOCAL_BACKEND_CLOSE_WAIT_S` | 15 秒 | 关闭实例时的等待时间；超时后转到后台继续关闭，重复请求不会重复发起清理 |

两项都必须是有限正数。

清理卡住时，资源尚未确认释放，程序不会另启新实例来替代。请按 GPU 提示条及下一节处理。

## 显卡与取消

### CUDA 不可用或显存不足

语音识别必须使用 CUDA，不会悄悄改用 CPU。先检查 NVIDIA 驱动，并关闭其他占用显卡的程序。程序最低要求 8 GB 物理显存。

请保留默认的 `ASR_BATCH_SIZE=auto`，程序会按显卡总/空闲显存、精度与分块上限通过显存公式直接计算批大小（最高 24），并在意外显存不足（OOM）时自动减半重试。目前没有更小的识别模型可换；调低批大小也解决不了驱动不可用或显存低于最低要求的问题。

### ASR 子进程连接中断

ASR 子进程在返回结果前断开时，程序会短暂等待它自然退出，记录 PID、退出码和最后收到的阶段消息。确认子进程已清理后，程序会重启一次并复用已写入的分块缓存。文本转写显示已完成的块数。已完成的 ASR 结果会先交给主进程，临时 WAV 随后清理。

排查时可对照任务的 `.run.log` 和 `logs/asr-worker-<PID>.log`。后者记录 worker 的异常、临时 WAV 清理进度，以及运行时能够捕捉的原生故障堆栈。

### “停止中”一直不结束

点击取消后，任务先显示“停止中”；正在执行的环节确认退出后，才会变为“已取消”。之后删除任务，才会清理它的临时目录。

取消本地翻译时，llama-server 可能还在生成，显存不一定立即释放。被取消过的服务实例不再接收新任务：正在使用它的其他任务会正常完成，之后的本地翻译任务会等旧实例退出后再启动新实例；等待期间可以删除任务来结束等待。API 翻译不受影响。

### GPU 未释放提示条

识别子进程或本地翻译服务未能确认退出时，任务列表上方会出现提示条，并暂停新的 GPU 任务和本地模型加载。程序会定时重试清理，你也可以点“重新检查”或“重试清理”。

“重试清理”只处理已经卡住的资源，不会停止其他正常运行的任务。提示条只在资源确认退出后才会消失，其他请求成功不能代替退出确认。

如果提示条一直存在，请结合 `logs/` 日志确认残留进程的状态并手动处理；程序不会强行终止 Python 线程。等待超时不代表显卡资源已经释放。

## 结果与恢复

### 保存失败后怎么继续

任务正常结束时先显示“保存中”，结果确认写入磁盘后才显示“完成”。

| 状态 | 操作 | 是否重新识别或翻译 |
| --- | --- | --- |
| 保存失败 | 点击“重试保存”，重新写入任务状态 | 否 |
| 产物提交失败 | 点击“重试提交”，提交已保存的整套结果 | 否 |
| 字幕导出失败 | 点击“重试导出”，把已保存的字幕导出到目标位置 | 否 |

重试前先排除磁盘空间不足、目录无写入权限或文件被占用等问题。这些重试不需要重新配置 API。已提交的结果仍可从任务卡下载；文件损坏时会明确报错，需要先恢复文件再重试。

同一输入和设置重复提交时，正在运行或等待保存的任务会被复用。如果不同输入或设置会写到同一个正在使用的字幕路径，请处理现有任务或选择其他输出目录。旧任务已提交的结果仍可从各自的任务卡下载。如果取消请求晚于整套结果提交，这次已完成的结果会保留。

### 自动化任务

`POST /api/jobs` 按提交顺序返回 `ids`。正在排队、运行或等待保存的相同任务会返回现有 ID，同一批次重复的条目也可返回同一个 ID。不同任务争用同一字幕输出路径时，整批请求返回 HTTP 409。响应的 `job_id` 指向已存在的任务，批次内部的冲突返回 `null`。已完成的任务可以再次提交。

使用 `GET /api/jobs/{job_id}` 查询状态和 `subtitle_cue_count`。`done` 且条数为 `0` 表示处理结束，生成的字幕没有条目。`null` 表示条数尚未记录，较早的产物也可能返回这个值。

遇到 `save_failed`、`publish_failed` 或 `export_failed` 时，排除写盘问题后调用 `POST /api/jobs/{job_id}/retry` 恢复该任务。恢复完成后自动清理本次任务的临时文件。

### 目录用途与清理

| 位置 | 内容 | 说明 |
| --- | --- | --- |
| 所选输出目录 | 导出的 SRT；留空时为视频所在目录 | 播放器直接读取 |
| `config/config.json` | 本机设置（API Key、代理、覆盖参数） | 升级保留，独立存放 |
| `logs/` | 进程日志与各任务 `.run.log` | 便于排查问题 |
| `outputs/` | 每次运行的整套不可变正式结果（SRT、中间 JSON 与同步拷贝的 `.run.log`） | 删任务卡不删除，不自动清理 |
| `cache/` | 任务状态（`cache/web/`）与跨任务可复用缓存（`asr/`、`llm/`、`torch/`、`uv/`） | 升级保留，分享日志无需打包 |
| `models/` | 已下载的模型与本地引擎（`models/runtime/`） | 升级保留 |
| `.venv/`、`.python/` | Python 运行环境与独立解释器 | 升级保留 |
| `tmp/` | 纯临时工作区（运行中与失败任务的 `tmp/jobs/`、切片 `tmp/chunks/`、实例锁、升级缓存 `tmp/updates/` 与升级遗留） | 随时整目录清空也不影响配置、历史与环境 |

删除任务卡会保留正式结果和已导出的 SRT。`outputs/` 不会自动清理；需要腾出空间时，先备份所需结果并关闭程序，再删除不需要的结果目录。

任务成功后会立即清理该任务的临时目录与提取 WAV，保留模型和 `cache/` 中跨任务复用的识别缓存；**失败任务会自动保留 `tmp/jobs/<job_id>/audio/` 中的提取 WAV**，方便重试复用或单独截取音频复现问题（在界面删除该任务卡后会一并清理）。同一程序目录打开多个窗口时，只有最后关闭的窗口执行退出清理，以免影响其他任务；清理期间启动新窗口会先等待，最多 20 秒，并在命令行窗口中说明原因。

## 配置校验

`config/config.json` 只需保存本机设置（JSON 对象格式），未设置的选项使用程序内置默认值；进程环境变量仍优先于 `config/config.json`。保存失败会保留原来的运行配置；保存完成时也不会覆盖保存期间新输入的内容。

一般的数字和开关设置留空时使用默认值；无法解析时也使用默认值，并提示字段名。数字越界时，上限类设置限制到边界，速率类设置恢复默认值。不接受 `inf`、`nan`；整数设置不能填写 `4.5` 这类小数。

启动时会集中列出已读取配置的问题，例如：

```text
[WARN] 配置项 SUBTITLE_LINGER_S='-1'：低于允许的最小值 0，已按该边界处理
```

只在任务中读取的设置会在首次读取时提示；改回合法值后，下次读取时提示即消失。

只接受固定取值的设置填错时会直接报错，例如 `SUBTITLE_LAYOUT_ENGINE`、`SUBTITLE_TIMING_MODEL`、`ASR_CHUNK_PAUSE_READING`。按报错中的字段名和允许值修改后再运行。

## 从源码运行

从源码运行需要 Git、uv、满足 `pyproject.toml` 版本约束的 Python，以及 FFmpeg Shared。Windows 上要确保 Shared 版在 `PATH` 中，且静态版不能排在它前面：TorchCodec 需要加载 FFmpeg 的共享 DLL。

```powershell
winget install --id Gyan.FFmpeg.Shared --exact

git clone https://github.com/jaykwok/jav-sub.git
cd jav-sub

$env:PYTHONIOENCODING = "utf-8"
uv venv
uv sync
uv run --no-sync python launcher.py
```

浏览器默认打开 `http://127.0.0.1:2233`。端口被占用时会自动改用下一个可用端口，实际地址见启动日志。

请用 uv 按项目锁定配置安装依赖，不要用全局 Python 或逐个 `pip install` 拼装环境，以免装成 CPU 版 PyTorch。在 git 工作副本中，发布启动器的自检只报告环境状态、不会改动 `.venv`；同步依赖请直接使用 `uv sync`。

## 反馈问题

请先在 [GitHub Issues](https://github.com/jaykwok/jav-sub/issues) 反馈，说明所用模式、操作步骤和报错信息。常用日志位于 `logs/`（不含 `config/config.json`）：

- `logs/app-<PID>.log`（及原生故障日志 `app-<PID>-fault.log`）：主程序、Web 服务与后台任务日志（按进程分文件，多窗口同时运行互不干扰）
- `logs/bootstrap.log`：启动器与首次环境自检日志
- `logs/jobs/<job_id>/*.run.log`（及失败时保留的 `*.timings.json`）：单任务各阶段日志；已完成任务的 `outputs/` 产物目录内也同步保留了一份刷新后的 `.run.log`
- `logs/asr-worker-<PID>.log`：ASR GPU 子进程日志（含原生 stderr 与故障堆栈）
- `logs/llamacpp_server.log`：本地翻译服务日志
- 失败任务的提取音频默认保留在 `tmp/jobs/<job_id>/audio/`，便于截短复现

需要附日志时，提供对应的 `.run.log` / `app-<PID>.log` 片段，以及能复现问题的最小字幕或音频片段。

反馈时请勿附带 API Key 或 `config/config.json`。
