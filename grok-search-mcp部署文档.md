# Claude Code / Codex grok-search MCP 全新部署文档（reasoning xhigh）

本文档给 AI Agent 使用，用于从零部署一套统一的 `grok-search-mcp`，让 Claude Code 和 Codex 共用同一个 MCP server 入口。

当前推荐模型示例是：

```text
grok-4.20-0309-reasoning-super
```

但模型名不写死在魔改代码里，只通过 `GROK_MODEL` 和 `~/.config/grok-search/config.json` 控制。后续切换到 `grok-4.3-beta` 或其他 reasoning 模型时，只改配置，不改 `dist/providers/grok.js`。

## 1. 最终能力

部署完成后应支持：

- `get_config_info`：查看 Grok Search MCP 配置和连接状态。
- `switch_model`：切换默认模型。
- `web_search`：搜索和资料整理。
- `web_fetch`：单页抓取，默认快速摘要 `mode=summary`。
- `web_fetch_batch`：多 URL 并发抓取，避免 AI 串行调用多个 `web_fetch`。

推荐调用规则：

1. 默认资料整理使用 `web_search(fetch_strategy="single_response")`。
2. 多个明确 URL 核实时，使用 `web_fetch_batch(mode="summary", concurrency=3~5)`。
3. 只有需要完整网页 Markdown 时，才使用 `web_fetch(mode="full")`。
4. 不要让 AI 连续串行调用多个 `web_fetch`。
5. 不要默认使用 `web_search(fetch_strategy="multi_request_parallel")`。

## 2. 核心原则

- `grok-search-mcp` 代码只部署和魔改一次。
- Claude Code 和 Codex 都是 MCP 客户端，需要分别写入 MCP 配置。
- 两边 MCP 配置的 `command` 和 `args` 指向同一个固定 `dist/server.js`。
- 推荐使用稳定部署目录安装和魔改，不要魔改 npx 缓存目录。npx 缓存可能被清理或路径变化，容易导致 MCP 配置指向失效文件。
- 但稳定部署目录必须最终写入 MCP 客户端配置的 `command`/`args`，否则客户端仍然会读取旧路径。
- `GROK_API_KEY` 必须向用户单独索取，不要从项目 `.env`、Codex/Claude 模型 key、`~/.codex/auth.json` 或其他业务配置里复制。
- reasoning effort 只保留 `xhigh`，不保留其他 effort 档位配置。
- 模型名只作为环境变量，不写进代码判断。

### 2.1 Agent 执行总览

Agent 按本文档部署时，按下面顺序执行：

1. 向用户索取真实 `GROK_API_KEY`，不要自行从其他配置中复制。
2. 确认 Node.js 和 npm 可用。
3. 在稳定部署目录安装 `grok-search-mcp`。
4. 定位稳定部署目录里的 `dist/providers/grok.js` 和 `dist/server.js`。
5. 按第 9、10 节完成魔改。
6. 写入 `~/.config/grok-search/config.json`，只写模型名，不写 API key。
7. 配置 Claude Code 和/或 Codex MCP，`args` 指向稳定部署目录里的 `server.js`。
8. 如果用户使用 cc-switch，在 cc-switch 中确认同一个 MCP 条目已同步到目标客户端，并核对同步后的 `args`。
9. 提醒用户重启对应客户端。
10. 按第 16 节验证。
11. 按第 21 节输出部署完成后的用户提示。

执行过程中不要把真实 API key 写入项目文档、日志、临时文件或截图。需要记录配置时，只记录脱敏值。

## 3. 固定参数

### 3.1 API 地址

```text
https://www.micuapi.ai/v1
```

### 3.2 当前推荐模型

```text
grok-4.20-0309-reasoning-super
```

后续如果切换模型，只改下面这些位置的 `GROK_MODEL` 或 `model`：

- Claude Code MCP env
- Codex MCP env
- `~/.config/grok-search/config.json`

不要改魔改代码。

### 3.3 reasoning 和超时策略

reasoning 只保留 xhigh：

```text
GROK_INCLUDE_REASONING=true
GROK_REASONING_EFFORT=xhigh
```

xhigh 对应更深的多 agent 推理，单次请求耗时可能较长。Codex 工具调用存在约 120 秒等待上限，因此推荐：

```text
GROK_REQUEST_TIMEOUT_MS=85000
GROK_RETRY_COUNT=0
```

含义：

- 单个 `/responses` 请求最多等待 85 秒。
- 给 Codex 120 秒工具等待上限保留约 30 秒用于 MCP 进程、SSE 读取、JSON 处理和客户端传输。
- xhigh 不建议默认重试。一次 85 秒请求失败后直接返回错误，比自动重试后被 Codex 客户端 120 秒硬超时更可控。

可选调整：

- 网络较好、但 xhigh 经常 80 秒附近才返回：调到 `90000`。
- 经常遇到 Codex `timed out awaiting tools/call after 120s`：调到 `75000`。
- 不建议超过 `95000`，否则 MCP 自身超时和 Codex 客户端 120 秒上限之间的缓冲太小。
- 不建议低于 `60000`，否则 xhigh 复杂搜索很容易被本地提前 abort。

### 3.4 统一 MCP 环境变量

Claude Code 和 Codex 使用同一组 MCP env：

```text
GROK_API_URL=https://www.micuapi.ai/v1
GROK_API_KEY=sk-xxx
GROK_MODEL=grok-4.20-0309-reasoning-super
GROK_INCLUDE_REASONING=true
GROK_REASONING_EFFORT=xhigh
GROK_MAX_TOKENS=38400
GROK_REQUEST_TIMEOUT_MS=85000
GROK_RETRY_COUNT=0
GROK_SEARCH_FETCH_STRATEGY=single_response
GROK_SEARCH_AUTO_FETCH=false
GROK_SEARCH_FETCH_LIMIT=3
GROK_SEARCH_FETCH_CONCURRENCY=3
GROK_SEARCH_FETCH_MAX_CHARS=6000
GROK_FETCH_MODE=summary
GROK_FETCH_BATCH_CONCURRENCY=5
GROK_FETCH_BATCH_MAX_CHARS=4000
DEBUG=true
GROK_DEBUG=true
```

`sk-xxx` 是占位符。执行部署前，必须提示用户提供真实 `GROK_API_KEY`。

API key 写入位置：

- Claude Code：`~/.claude.json` 的 `mcpServers.grok-search.env.GROK_API_KEY`
- Codex：`~/.codex/config.toml` 的 `[mcp_servers.grok-search.env]` 下的 `GROK_API_KEY`

不要把 `GROK_API_KEY` 写入 `~/.config/grok-search/config.json`。

## 4. 准备环境

确认本机有 Node.js 和 npm：

```bash
node -v
npm -v
```

如果没有，先安装 Node.js LTS。

## 5. 稳定目录安装 grok-search-mcp

不要把 npx 缓存目录作为长期部署目录。推荐安装到稳定目录，然后魔改这个稳定目录里的编译产物。

Windows PowerShell：

```powershell
$DeployRoot = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified"
New-Item -ItemType Directory -Force $DeployRoot | Out-Null
Push-Location $DeployRoot
npm init -y
npm install grok-search-mcp@latest
Pop-Location
```

Linux/macOS Bash：

```bash
DEPLOY_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/grok-search-mcp-unified"
mkdir -p "$DEPLOY_ROOT"
cd "$DEPLOY_ROOT"
npm init -y
npm install grok-search-mcp@latest
```

升级包版本时，`npm install` 可能覆盖 `dist` 下的魔改文件。升级后必须重新检查第 9、10 节魔改是否仍然存在。

## 6. 定位稳定部署文件

Windows PowerShell：

```powershell
$DeployRoot = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified"
$ServerJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\server.js"
$GrokJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\providers\grok.js"
Test-Path $ServerJs
Test-Path $GrokJs
$ServerJs
$GrokJs
```

Linux/macOS Bash：

```bash
DEPLOY_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/grok-search-mcp-unified"
SERVER_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/server.js"
GROK_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/providers/grok.js"
test -f "$SERVER_JS" && printf '%s\n' "$SERVER_JS"
test -f "$GROK_JS" && printf '%s\n' "$GROK_JS"
```

找到 `server.js` 后，确认同目录存在：

```text
<grok-search-mcp目录>/dist/providers/grok.js
<grok-search-mcp目录>/dist/server.js
```

后续 Claude Code 和 Codex 都必须使用稳定部署目录里的同一个 `dist/server.js` 绝对路径。

## 7. 实施前备份

修改 MCP 编译产物或客户端配置前，先备份。备份目录固定为项目根目录下的 `grok-search备份/`。

Windows PowerShell：

```powershell
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$BackupDir = Join-Path $env:USERPROFILE "grok-search备份"
New-Item -ItemType Directory -Force $BackupDir | Out-Null

$DeployRoot = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified"
$ServerJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\server.js"
$GrokJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\providers\grok.js"

if (Test-Path $ServerJs) { Copy-Item $ServerJs (Join-Path $BackupDir "server.$Stamp.js") }
if (Test-Path $GrokJs) { Copy-Item $GrokJs (Join-Path $BackupDir "grok.$Stamp.js") }
if (Test-Path "$env:USERPROFILE\.claude.json") { Copy-Item "$env:USERPROFILE\.claude.json" (Join-Path $BackupDir "claude.$Stamp.json") }
if (Test-Path "$env:USERPROFILE\.codex\config.toml") { Copy-Item "$env:USERPROFILE\.codex\config.toml" (Join-Path $BackupDir "codex-config.$Stamp.toml") }
```

Linux/macOS Bash：

```bash
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="$HOME/grok-search备份"
mkdir -p "$BACKUP_DIR"

DEPLOY_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/grok-search-mcp-unified"
SERVER_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/server.js"
GROK_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/providers/grok.js"

test -f "$SERVER_JS" && cp "$SERVER_JS" "$BACKUP_DIR/server.$STAMP.js"
test -f "$GROK_JS" && cp "$GROK_JS" "$BACKUP_DIR/grok.$STAMP.js"
test -f "$HOME/.claude.json" && cp "$HOME/.claude.json" "$BACKUP_DIR/claude.$STAMP.json"
test -f "$HOME/.codex/config.toml" && cp "$HOME/.codex/config.toml" "$BACKUP_DIR/codex-config.$STAMP.toml"
```

备份文件可能包含 API key。不要把 `grok-search备份/` 里的配置备份发给他人或同步到外部。

## 8. 固定模型配置

`grok-search-mcp` 会读取 `~/.config/grok-search/config.json`。为避免旧模型覆盖环境变量，写入当前模型。这里只写模型名，不写 API key。

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.config\grok-search" | Out-Null
Set-Content -Path "$env:USERPROFILE\.config\grok-search\config.json" -Value '{"model":"grok-4.20-0309-reasoning-super"}'
```

Linux/macOS Bash：

```bash
mkdir -p ~/.config/grok-search
printf '{"model":"grok-4.20-0309-reasoning-super"}\n' > ~/.config/grok-search/config.json
```

后续换模型时，只替换 JSON 里的 `model` 值。

## 9. 修改 dist/providers/grok.js

### 9.1 搜索策略

`search()` 必须支持三种策略：

- `single_response`：默认；一次 `/responses` 内完成搜索、阅读和总结。
- `none`：只搜索，不要求阅读详情。
- `multi_request_parallel`：搜索后并发抓取页面，仅在明确需要搜索后自动抓取时使用。

核心逻辑：

```js
const fetchStrategy = options.fetchStrategy || process.env.GROK_SEARCH_FETCH_STRATEGY || "single_response";
let researchPrompt = "";

if (fetchStrategy === "single_response") {
  const readLimit = Number(options.fetchLimit ?? process.env.GROK_SEARCH_FETCH_LIMIT ?? Math.min(3, maxResults || 3));
  researchPrompt = `\n\nResearch mode: use web_search in this same response to search, open/read the most authoritative top ${readLimit} results when useful, and synthesize the findings. Do not ask the caller to fetch URLs separately. Return a JSON object with {"strategy":"single_response","request_count":1,"results":[...],"analysis":"..."}. Each result should include title, url, description, source_type, key_findings, and confidence when available.`;
}
```

执行搜索后：

```js
if (fetchStrategy === "multi_request_parallel") {
  return this.enrichSearchResults(results, ctx, { ...options, autoFetch: true }, maxResults);
}

if (fetchStrategy === "single_response") {
  return this.wrapSingleResponseSearch(results);
}

return results;
```

`single_response` 必须由 MCP 代码层稳定包装，不要只依赖模型按 prompt 自觉返回 `strategy/request_count/analysis`。即使模型返回普通数组，也要包装成统一 envelope：

```js
parseJsonOutput(text) {
  const trimmed = text.trim().replace(/^```json\s*/i, "").replace(/```$/i, "").trim();
  try {
    return JSON.parse(trimmed);
  } catch (_error) {
    const starts = [
      trimmed.indexOf("{"),
      trimmed.indexOf("["),
    ].filter((index) => index >= 0);
    if (!starts.length) {
      return null;
    }
    const start = Math.min(...starts);
    for (let end = trimmed.length; end > start; end--) {
      try {
        return JSON.parse(trimmed.slice(start, end));
      } catch (_innerError) {
        // Keep shrinking until a valid JSON object or array is found.
      }
    }
    return null;
  }
}

wrapSingleResponseSearch(resultsText) {
  const parsed = this.parseJsonOutput(resultsText);

  if (parsed && !Array.isArray(parsed) && parsed.strategy === "single_response") {
    return JSON.stringify({
      strategy: "single_response",
      request_count: parsed.request_count || 1,
      results: Array.isArray(parsed.results) ? parsed.results : [],
      analysis: parsed.analysis || "",
      ...parsed,
    }, null, 2);
  }

  if (Array.isArray(parsed)) {
    return JSON.stringify({
      strategy: "single_response",
      request_count: 1,
      results: parsed,
      analysis: "",
    }, null, 2);
  }

  if (parsed && !Array.isArray(parsed) && Array.isArray(parsed.results)) {
    return JSON.stringify({
      strategy: "single_response",
      request_count: parsed.request_count || 1,
      results: parsed.results,
      analysis: parsed.analysis || parsed.summary || "",
    }, null, 2);
  }

  return JSON.stringify({
    strategy: "single_response",
    request_count: 1,
    results: [],
    analysis: resultsText,
  }, null, 2);
}
```

### 9.2 /responses + SSE 解析 + xhigh reasoning

`executeStream()` 必须：

1. 请求 `${this.apiUrl}/responses`。
2. 使用 `input: [{ role: "user", content: ... }]`。
3. 设置 `stream: true`。
4. 根据 `payload.toolType` 选择 `{ type: "web_search" }` 或 `{ type: "web_fetch" }`。
5. 使用 `response.text()`，不要用 `response.json()`。
6. 解析 SSE 中的 `data:` 行时，只保留最终回答文本：优先拼接 `response.output_text.delta`，其次使用 `response.output_text.done` / `response.completed` 中 message 的 `output_text`。不要把 reasoning、tool 调用日志、`browse_page`、`chatroom_send`、`[Grok]` 内部协作日志拼进结果。
7. 增加 `AbortController` 超时控制。
8. reasoning effort 固定从 `GROK_REASONING_EFFORT` 读取，默认 `xhigh`。
9. 不要在代码里判断具体模型名。

推荐请求体：

```js
const tool = payload.toolType === "web_fetch" ? { type: "web_fetch" } : { type: "web_search" };
if (payload.maxResults && tool.type === "web_search") {
  tool.max_results = payload.maxResults || 20;
}

const includeReasoning = process.env.GROK_INCLUDE_REASONING !== "false";
const reasoningEffort = process.env.GROK_REASONING_EFFORT || "xhigh";
const maxTokens = Number(process.env.GROK_MAX_TOKENS || payload.max_tokens || payload.maxTokens);

const responsesPayload = {
  model: payload.model,
  input: [
    {
      role: "user",
      content: `${instructions}\n\n${input}`.trim(),
    },
  ],
  stream: true,
  tools: [tool],
  ...(includeReasoning ? {
    reasoning: { effort: reasoningEffort },
    include_reasoning: true,
  } : {}),
  max_tokens: Number.isFinite(maxTokens) && maxTokens >= 38400 ? maxTokens : 38400,
};
```

推荐超时包装：

```js
const requestTimeoutMs = Number(process.env.GROK_REQUEST_TIMEOUT_MS || 85000);
const useTimeout = Number.isFinite(requestTimeoutMs) && requestTimeoutMs > 0;
const controller = useTimeout ? new AbortController() : null;
const timeout = controller
  ? setTimeout(() => controller.abort(), requestTimeoutMs)
  : null;

let response;
let responseText;
try {
  response = await fetch(`${this.apiUrl}/responses`, {
    method: "POST",
    headers,
    body: JSON.stringify(responsesPayload),
    signal: controller?.signal,
  });
  responseText = await response.text();
} catch (error) {
  if (error?.name === "AbortError") {
    throw new Error(`Grok API request timed out after ${requestTimeoutMs}ms`);
  }
  throw error;
} finally {
  if (timeout) {
    clearTimeout(timeout);
  }
}
```

读取 `responseText` 后必须检查 HTTP 状态并调用下面的 SSE 解析 helper：

```js
if (!response.ok) {
  throw new Error(`Grok API error: ${response.status} - ${responseText}`);
}

return this.parseResponsesSse(responseText);
```

推荐 SSE 解析 helper。这个版本只提取最终回答文本，避免把 Grok Responses 的内部多 agent 日志、`browse_page`、`chatroom_send` 和 `[Grok]` 协作输出混入工具结果：

```js
extractFinalMessageText(container) {
  let content = "";
  const outputs = Array.isArray(container?.output)
    ? container.output
    : Array.isArray(container?.response?.output)
      ? container.response.output
      : [];

  for (const output of outputs) {
    if (output?.type && output.type !== "message") {
      continue;
    }
    if (!Array.isArray(output?.content)) {
      continue;
    }
    for (const item of output.content) {
      if (item?.type && item.type !== "output_text" && item.type !== "text") {
        continue;
      }
      if (typeof item?.text === "string") {
        content += item.text;
      }
      if (typeof item?.output_text === "string") {
        content += item.output_text;
      }
    }
  }

  return content;
}

extractLegacyText(parsed) {
  let content = "";
  if (typeof parsed?.output_text === "string") {
    content += parsed.output_text;
  }
  if (typeof parsed?.text === "string") {
    content += parsed.text;
  }
  if (Array.isArray(parsed?.choices)) {
    for (const choice of parsed.choices) {
      const delta = choice?.delta || {};
      if (typeof delta.content === "string") {
        content += delta.content;
      }
      if (typeof choice?.message?.content === "string") {
        content += choice.message.content;
      }
    }
  }
  return content;
}

parseResponsesSse(responseText) {
  let deltaContent = "";
  let doneContent = "";
  let completedContent = "";

  for (const rawLine of responseText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line.startsWith("data:")) {
      continue;
    }
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") {
      continue;
    }
    try {
      const parsed = JSON.parse(data);
      if (parsed?.type === "response.output_text.delta" && typeof parsed.delta === "string") {
        deltaContent += parsed.delta;
      } else if (parsed?.type === "response.output_text.done") {
        doneContent = parsed.text || parsed.output_text || doneContent;
      } else if (parsed?.type === "response.completed") {
        completedContent = this.extractFinalMessageText(parsed);
      }
    } catch (_error) {
      // Ignore non-JSON SSE lines.
    }
  }

  if (deltaContent) {
    return deltaContent;
  }
  if (doneContent) {
    return doneContent;
  }
  if (completedContent) {
    return completedContent;
  }

  try {
    const parsed = JSON.parse(responseText);
    return this.extractFinalMessageText(parsed) || this.extractLegacyText(parsed) || responseText;
  } catch (_error) {
    return responseText;
  }
}
```

### 9.3 retry 设置

xhigh 请求本身较慢，不建议单次工具调用内自动重试。若现有代码有 `retryWithContext()`，应让重试次数可由环境变量控制：

```js
const retryCount = Number(process.env.GROK_RETRY_COUNT ?? 0);
const retryOptions = {
  retries: Number.isFinite(retryCount) ? retryCount : 0,
};
```

如果原包的 retry helper 使用字段名不是 `retries`，按原 helper 的实际参数名适配。核心目标是：`GROK_RETRY_COUNT=0` 时不重试。

### 9.4 web_fetch 快速摘要模式

`fetch()` 增加 `mode`：

```js
async fetch(url, ctx, options = {}) {
  const mode = options.mode || process.env.GROK_FETCH_MODE || "summary";
  const userPrompt = mode === "full"
    ? `${url}\nFetch the content of this webpage and return it in structured Markdown format`
    : `${url}\nFetch this webpage quickly. Do not reproduce full Markdown. Return a compact JSON object with title, url, source_type, key_facts, dates, license, model_size, requirements, and a short summary when available.`;

  const payload = {
    model: this.model,
    messages: [
      {
        role: "system",
        content: mode === "full" ? fetchPrompt : "You are a fast web research extractor. Prefer concise factual extraction over full-page Markdown reproduction.",
      },
      {
        role: "user",
        content: userPrompt,
      },
    ],
    toolType: "web_fetch",
    stream: true,
  };

  return retryWithContext(() => this.executeStream(headers, payload, ctx), retryOptions);
}
```

### 9.5 web_fetch_batch 并发抓取

新增 `fetchBatch()`，用已有 `mapConcurrent()` 并发调用 `fetch()`：

```js
async fetchBatch(urls, ctx, options = {}) {
  const uniqueUrls = [...new Set(urls)].filter((url) => typeof url === "string" && /^https?:\/\//i.test(url));
  const concurrency = Math.max(1, Math.min(Number(options.concurrency ?? process.env.GROK_FETCH_BATCH_CONCURRENCY ?? 5), uniqueUrls.length));
  const maxContentChars = Number(options.maxContentChars ?? process.env.GROK_FETCH_BATCH_MAX_CHARS ?? 4000);
  const startedAt = Date.now();

  const results = await this.mapConcurrent(uniqueUrls, concurrency, async (url) => {
    const itemStartedAt = Date.now();
    try {
      const content = await this.fetch(url, ctx, { mode: options.mode || "summary" });
      return {
        url,
        fetched: true,
        elapsed_ms: Date.now() - itemStartedAt,
        content: this.truncateText(content, maxContentChars),
      };
    } catch (error) {
      return {
        url,
        fetched: false,
        elapsed_ms: Date.now() - itemStartedAt,
        error: error instanceof Error ? error.message : "Unknown fetch error",
      };
    }
  });

  return JSON.stringify({
    strategy: "parallel_batch",
    request_count: uniqueUrls.length,
    concurrency,
    elapsed_ms: Date.now() - startedAt,
    results,
  }, null, 2);
}
```

注意：先去重和过滤 URL，再计算 concurrency，避免 `urls.length` 和实际请求数量不一致。

## 10. 修改 dist/server.js

### 10.1 web_search schema

`web_search` 暴露：

```js
fetch_strategy: z.enum(["single_response", "none", "multi_request_parallel"])
  .optional()
  .default("single_response"),
auto_fetch: z.boolean().optional().default(false),
fetch_limit: z.number().optional().default(3),
fetch_concurrency: z.number().optional().default(3),
fetch_max_chars: z.number().optional().default(6000),
```

handler 中：

```js
const effectiveFetchStrategy = auto_fetch ? "multi_request_parallel" : fetch_strategy;
const results = await provider.search(query, platform, min_results, max_results, ctx, {
  fetchStrategy: effectiveFetchStrategy,
  autoFetch: auto_fetch,
  fetchLimit: fetch_limit,
  concurrency: fetch_concurrency,
  maxContentChars: fetch_max_chars,
});
```

### 10.2 web_fetch schema

`web_fetch` 默认 summary：

```js
server.registerTool("web_fetch", {
  description: `Fetches one webpage. Default mode is fast summary extraction, not full Markdown.

Use mode="summary" for speed when researching multiple sources.
Use mode="full" only when exact full-page Markdown is required.`,
  inputSchema: {
    url: z.string().describe("The URL of the web page to fetch"),
    mode: z.enum(["summary", "full"]).optional().default("summary"),
  },
}, async ({ url, mode = "summary" }) => {
  const results = await provider.fetch(url, ctx, { mode });
  return { content: [{ type: "text", text: results }] };
});
```

### 10.3 新增 web_fetch_batch

```js
server.registerTool("web_fetch_batch", {
  description: `Fetches multiple webpages concurrently. Prefer this over calling web_fetch repeatedly when speed matters.

Default mode="summary" extracts compact factual JSON from each URL. Use mode="full" only for full Markdown pages.`,
  inputSchema: {
    urls: z.array(z.string()).describe("URLs to fetch concurrently"),
    mode: z.enum(["summary", "full"]).optional().default("summary"),
    concurrency: z.number().optional().default(5),
    max_chars: z.number().optional().default(4000),
  },
}, async ({ urls, mode = "summary", concurrency = 5, max_chars = 4000 }) => {
  const results = await provider.fetchBatch(urls, ctx, {
    mode,
    concurrency,
    maxContentChars: max_chars,
  });
  return { content: [{ type: "text", text: results }] };
});
```

## 11. 固定启动路径

魔改完成后，不要把 MCP 启动命令写成：

```bash
npx -y grok-search-mcp
```

应固定为：

```bash
node <稳定部署目录>/node_modules/grok-search-mcp/dist/server.js
```

Claude Code 和 Codex 都指向同一个 `server.js`。这样只需要维护一份魔改代码。

## 12. 配置 Claude Code

推荐使用 CLI 写入。

Windows PowerShell：

```powershell
claude mcp remove grok-search -s local
$ServerJs = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified\node_modules\grok-search-mcp\dist\server.js"

claude mcp add grok-search -s local `
  -e GROK_API_URL=https://www.micuapi.ai/v1 `
  -e GROK_API_KEY=sk-xxx `
  -e GROK_MODEL=grok-4.20-0309-reasoning-super `
  -e GROK_INCLUDE_REASONING=true `
  -e GROK_REASONING_EFFORT=xhigh `
  -e GROK_MAX_TOKENS=38400 `
  -e GROK_REQUEST_TIMEOUT_MS=85000 `
  -e GROK_RETRY_COUNT=0 `
  -e GROK_SEARCH_FETCH_STRATEGY=single_response `
  -e GROK_SEARCH_AUTO_FETCH=false `
  -e GROK_SEARCH_FETCH_LIMIT=3 `
  -e GROK_SEARCH_FETCH_CONCURRENCY=3 `
  -e GROK_SEARCH_FETCH_MAX_CHARS=6000 `
  -e GROK_FETCH_MODE=summary `
  -e GROK_FETCH_BATCH_CONCURRENCY=5 `
  -e GROK_FETCH_BATCH_MAX_CHARS=4000 `
  -e DEBUG=true `
  -e GROK_DEBUG=true `
  -- node $ServerJs
```

Linux/macOS Bash：

```bash
claude mcp remove grok-search -s local

claude mcp add grok-search -s local \
  -e GROK_API_URL=https://www.micuapi.ai/v1 \
  -e GROK_API_KEY=sk-xxx \
  -e GROK_MODEL=grok-4.20-0309-reasoning-super \
  -e GROK_INCLUDE_REASONING=true \
  -e GROK_REASONING_EFFORT=xhigh \
  -e GROK_MAX_TOKENS=38400 \
  -e GROK_REQUEST_TIMEOUT_MS=85000 \
  -e GROK_RETRY_COUNT=0 \
  -e GROK_SEARCH_FETCH_STRATEGY=single_response \
  -e GROK_SEARCH_AUTO_FETCH=false \
  -e GROK_SEARCH_FETCH_LIMIT=3 \
  -e GROK_SEARCH_FETCH_CONCURRENCY=3 \
  -e GROK_SEARCH_FETCH_MAX_CHARS=6000 \
  -e GROK_FETCH_MODE=summary \
  -e GROK_FETCH_BATCH_CONCURRENCY=5 \
  -e GROK_FETCH_BATCH_MAX_CHARS=4000 \
  -e DEBUG=true \
  -e GROK_DEBUG=true \
  -- node "$HOME/.local/share/grok-search-mcp-unified/node_modules/grok-search-mcp/dist/server.js"
```

如果需要手动编辑，目标文件是：

```text
~/.claude.json
```

配置结构：

```json
{
  "mcpServers": {
    "grok-search": {
      "type": "stdio",
      "command": "node",
      "args": [
        "<ABS_SERVER_JS>"
      ],
      "env": {
        "GROK_API_URL": "https://www.micuapi.ai/v1",
        "GROK_API_KEY": "sk-xxx",
        "GROK_MODEL": "grok-4.20-0309-reasoning-super",
        "GROK_INCLUDE_REASONING": "true",
        "GROK_REASONING_EFFORT": "xhigh",
        "GROK_MAX_TOKENS": "38400",
        "GROK_REQUEST_TIMEOUT_MS": "85000",
        "GROK_RETRY_COUNT": "0",
        "GROK_SEARCH_FETCH_STRATEGY": "single_response",
        "GROK_SEARCH_AUTO_FETCH": "false",
        "GROK_SEARCH_FETCH_LIMIT": "3",
        "GROK_SEARCH_FETCH_CONCURRENCY": "3",
        "GROK_SEARCH_FETCH_MAX_CHARS": "6000",
        "GROK_FETCH_MODE": "summary",
        "GROK_FETCH_BATCH_CONCURRENCY": "5",
        "GROK_FETCH_BATCH_MAX_CHARS": "4000",
        "DEBUG": "true",
        "GROK_DEBUG": "true"
      }
    }
  }
}
```

## 13. 配置 Codex

推荐使用 CLI 写入。

Windows PowerShell：

```powershell
codex mcp remove grok-search
$ServerJs = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified\node_modules\grok-search-mcp\dist\server.js"

codex mcp add grok-search `
  --env GROK_API_URL=https://www.micuapi.ai/v1 `
  --env GROK_API_KEY=sk-xxx `
  --env GROK_MODEL=grok-4.20-0309-reasoning-super `
  --env GROK_INCLUDE_REASONING=true `
  --env GROK_REASONING_EFFORT=xhigh `
  --env GROK_MAX_TOKENS=38400 `
  --env GROK_REQUEST_TIMEOUT_MS=85000 `
  --env GROK_RETRY_COUNT=0 `
  --env GROK_SEARCH_FETCH_STRATEGY=single_response `
  --env GROK_SEARCH_AUTO_FETCH=false `
  --env GROK_SEARCH_FETCH_LIMIT=3 `
  --env GROK_SEARCH_FETCH_CONCURRENCY=3 `
  --env GROK_SEARCH_FETCH_MAX_CHARS=6000 `
  --env GROK_FETCH_MODE=summary `
  --env GROK_FETCH_BATCH_CONCURRENCY=5 `
  --env GROK_FETCH_BATCH_MAX_CHARS=4000 `
  --env DEBUG=true `
  --env GROK_DEBUG=true `
  -- node $ServerJs
```

Linux/macOS Bash：

```bash
codex mcp remove grok-search

codex mcp add grok-search \
  --env GROK_API_URL=https://www.micuapi.ai/v1 \
  --env GROK_API_KEY=sk-xxx \
  --env GROK_MODEL=grok-4.20-0309-reasoning-super \
  --env GROK_INCLUDE_REASONING=true \
  --env GROK_REASONING_EFFORT=xhigh \
  --env GROK_MAX_TOKENS=38400 \
  --env GROK_REQUEST_TIMEOUT_MS=85000 \
  --env GROK_RETRY_COUNT=0 \
  --env GROK_SEARCH_FETCH_STRATEGY=single_response \
  --env GROK_SEARCH_AUTO_FETCH=false \
  --env GROK_SEARCH_FETCH_LIMIT=3 \
  --env GROK_SEARCH_FETCH_CONCURRENCY=3 \
  --env GROK_SEARCH_FETCH_MAX_CHARS=6000 \
  --env GROK_FETCH_MODE=summary \
  --env GROK_FETCH_BATCH_CONCURRENCY=5 \
  --env GROK_FETCH_BATCH_MAX_CHARS=4000 \
  --env DEBUG=true \
  --env GROK_DEBUG=true \
  -- node "$HOME/.local/share/grok-search-mcp-unified/node_modules/grok-search-mcp/dist/server.js"
```

如果需要手动编辑，目标文件是：

```text
~/.codex/config.toml
```

Windows 上通常是：

```text
%USERPROFILE%\.codex\config.toml
```

配置结构：

```toml
[mcp_servers.grok-search]
type = "stdio"
command = "node"
args = ['<ABS_SERVER_JS>']

[mcp_servers.grok-search.env]
DEBUG = "true"
GROK_API_KEY = "sk-xxx"
GROK_API_URL = "https://www.micuapi.ai/v1"
GROK_DEBUG = "true"
GROK_FETCH_BATCH_CONCURRENCY = "5"
GROK_FETCH_BATCH_MAX_CHARS = "4000"
GROK_FETCH_MODE = "summary"
GROK_INCLUDE_REASONING = "true"
GROK_MAX_TOKENS = "38400"
GROK_MODEL = "grok-4.20-0309-reasoning-super"
GROK_REASONING_EFFORT = "xhigh"
GROK_REQUEST_TIMEOUT_MS = "85000"
GROK_RETRY_COUNT = "0"
GROK_SEARCH_AUTO_FETCH = "false"
GROK_SEARCH_FETCH_CONCURRENCY = "3"
GROK_SEARCH_FETCH_LIMIT = "3"
GROK_SEARCH_FETCH_MAX_CHARS = "6000"
GROK_SEARCH_FETCH_STRATEGY = "single_response"
```

Windows 路径建议在 TOML 里使用单引号：

```toml
args = ['C:\Users\<用户名>\AppData\Local\grok-search-mcp-unified\node_modules\grok-search-mcp\dist\server.js']
```

这样反斜杠不会被 TOML 当作转义字符处理。

如果已经用 `codex mcp add` 添加过，优先用下面命令检查，不要重复写多个同名配置段：

```bash
codex mcp get grok-search
```

## 14. cc-switch 使用方式

如果使用 cc-switch 管理 MCP：

1. 先确认第 6 节得到的 `server.js` 绝对路径真实存在。
2. 在 cc-switch 中只创建一个 `grok-search` MCP 条目。
3. `transport/type` 选择 `stdio`。
4. `command` 填 `node`。
5. `args` 填第 6 节得到的 `server.js` 绝对路径，不要填 npx 命令，也不要填旧缓存路径。
6. `env` 填第 3.4 节的统一环境变量。
7. 同时打开 Claude 和 Codex 的应用开关。

这样 cc-switch 会把同一个 MCP 条目同步到：

```text
cc-switch 自身配置源: ~/.cc-switch/cc-switch.db
Claude Code: ~/.claude.json
Codex: ~/.codex/config.toml
```

关键点：

- cc-switch 管理的是 MCP 客户端配置，不会自动寻找新的 `server.js`。
- 最终 Claude Code / Codex 启动 MCP 时，读取的是同步后的客户端配置里的 `command` 和 `args`。
- 如果更换了部署目录或重新安装包，必须回到 cc-switch 更新 `args`，再确认 `~/.claude.json` 和 `~/.codex/config.toml` 已同步。
- 不要在 Claude 和 Codex 里分别维护两份不同的 `grok-search` 魔改代码。两边只需要分别有 MCP 客户端配置，代码入口共用同一个 `server.js`。

## 15. 重启要求

修改以下内容后必须重启对应 Agent：

- `~/.claude.json`
- `~/.codex/config.toml`
- `dist/providers/grok.js`
- `dist/server.js`

原因：

- Claude Code / Codex 启动时读取 MCP 配置。
- MCP server 是已经启动的 Node 进程，修改磁盘文件不会热更新。

## 16. 验证

### 16.1 检查 Claude Code MCP

```bash
claude mcp get grok-search
```

确认：

```text
Status: Connected
transport/type: stdio
command: node
args: .../node_modules/grok-search-mcp/dist/server.js
env: 包含 GROK_MODEL、GROK_REASONING_EFFORT、GROK_REQUEST_TIMEOUT_MS、GROK_SEARCH_FETCH_STRATEGY
```

### 16.2 检查 Codex MCP

```bash
codex mcp list
codex mcp get grok-search
```

确认：

```text
grok-search
  enabled: true
  transport/type: stdio
  command: node
  args: .../node_modules/grok-search-mcp/dist/server.js
  env: ..., GROK_MODEL=*****, GROK_REASONING_EFFORT=*****, GROK_REQUEST_TIMEOUT_MS=*****
```

Codex 会脱敏显示环境变量值，这是正常的。

### 16.3 检查 grok-search 自身配置

在 Claude Code 里通过 `mcp__grok-search__get_config_info` 工具调用（参数留空）：

```json
{}
```

目标工具：`grok-search.get_config_info`

成功特征：

- `config_status` 为 complete 或等价成功状态。
- `api_url` 为 `https://www.micuapi.ai/v1`。
- `model` 为当前 `GROK_MODEL`。
- API key 只显示脱敏内容。

### 16.4 资料整理搜索

优先测试 `single_response`：

```json
{
  "query": "Grok 4.3 beta latest model documentation reasoning effort",
  "platform": "Web",
  "min_results": 3,
  "max_results": 5,
  "fetch_strategy": "single_response",
  "fetch_limit": 3
}
```

目标工具：`grok-search.web_search`

成功特征：

- 返回顶层 JSON envelope，必须包含 `strategy="single_response"`、`request_count=1`、`results`、`analysis`。
- 不出现顶层 `fetched`。
- 不应出现 `[Grok]`、`browse_page`、`chatroom_send` 等内部协作日志；如果出现，说明 `parseResponsesSse()` 仍把 reasoning/tool 事件拼进了结果，需要回到第 9.2 修正。
- 一次请求内完成搜索、阅读和总结。
- 在 xhigh 下可接受 60~90 秒返回。

### 16.5 单页快速摘要

```json
{
  "url": "https://docs.x.ai/",
  "mode": "summary"
}
```

目标工具：`grok-search.web_fetch`

成功特征：

- 默认返回紧凑摘要，不是完整 Markdown。
- 输出不应出现 `[Grok]`、`browse_page`、`chatroom_send` 等内部协作日志。

### 16.6 多页并发抓取

```json
{
  "urls": [
    "https://docs.x.ai/docs/models",
    "https://docs.x.ai/developers/model-capabilities/text/reasoning",
    "https://docs.x.ai/developers/tools/web-search"
  ],
  "mode": "summary",
  "concurrency": 3,
  "max_chars": 800
}
```

目标工具：`grok-search.web_fetch_batch`

成功特征：

- 返回 `strategy=parallel_batch`。
- 返回 `concurrency=3`。
- 总耗时接近最慢单页，而不是所有页面耗时相加。
- 每个 `results[].content` 不应出现 `[Grok]`、`browse_page`、`chatroom_send` 等内部协作日志。

## 17. 120 秒超时处理

### 17.1 典型错误

Codex 里常见：

```text
Error: tool call error: tool call failed for `grok-search/web_search`
Caused by:
    timed out awaiting tools/call after 120s
```

这通常不是 MCP 没部署成功，而是 MCP 工具调用没有在 Codex 客户端等待上限内返回。

### 17.2 xhigh 推荐解法

1. `GROK_REQUEST_TIMEOUT_MS=85000`。
2. `GROK_RETRY_COUNT=0`。
3. 默认资料整理使用 `web_search(fetch_strategy="single_response")`。
4. 多 URL 核实时使用 `web_fetch_batch(mode="summary", concurrency=3~5)`。
5. 避免 `web_search(fetch_strategy="multi_request_parallel")`。
6. 修改后重启 Claude Code / Codex。

### 17.3 何时调整 GROK_REQUEST_TIMEOUT_MS

推荐值：

```text
GROK_REQUEST_TIMEOUT_MS=85000
```

调整建议：

- 结果常在 85 秒前被本地 abort，但 Codex 不报 120 秒：调到 `90000`。
- Codex 经常报 120 秒硬超时：调到 `75000`。
- 只做短查询、希望快速失败：调到 `60000`。
- 不建议超过 `95000`。

### 17.4 Cloudflare 524

日志中出现：

```text
Grok API error: 524
A timeout occurred
www.micuapi.ai
```

说明请求已经到达 Micu API 网关，但源站处理太久。

处理：

- 保持 `GROK_RETRY_COUNT=0`，不要在单次工具调用里自动重试。
- 降低 `max_results` 和 `fetch_limit`。
- 避免 `multi_request_parallel`。
- 对明确 URL 列表使用 `web_fetch_batch(mode="summary", concurrency=3~5)`。
- 必要时把 `GROK_REQUEST_TIMEOUT_MS` 从 `85000` 调到 `75000`，让失败更早返回。

## 18. 常见问题

### 18.1 invalid character ':' looking for beginning of value

原因：后端返回 `text/event-stream`，但代码用 `response.json()` 解析。

解决：用 `response.text()` 读取，再解析 SSE 的 `data:` 行。

### 18.2 输出混入 `[Grok]`、`browse_page`、`chatroom_send`

原因：`parseResponsesSse()` 把 Responses API 的 reasoning/tool 事件或内部多 agent 协作日志也拼进了结果。

解决：

- 回到第 9.2，使用只读取 `response.output_text.delta`、`response.output_text.done`、`response.completed` message `output_text` 的解析 helper。
- 不要把任意 SSE JSON 里的 `delta`、`text` 字段无条件拼接。
- 修改 `dist/providers/grok.js` 后必须重启 Claude Code / Codex。

### 18.3 `web_search(single_response)` 没有 `strategy`

原因：模型返回了普通结果数组，MCP 代码没有在 provider 层兜底包装。

解决：回到第 9.1，确认 `search()` 在 `fetchStrategy === "single_response"` 时调用 `wrapSingleResponseSearch(results)`，并且 `dist/providers/grok.js` 里存在 `parseJsonOutput()` / `wrapSingleResponseSearch()`。

### 18.4 搜索后还是很慢

检查日志。

Windows PowerShell：

```powershell
Get-Content "$env:USERPROFILE\.config\grok-search\logs\grok_search_YYYYMMDD.log" -Tail 120
```

如果出现：

```text
Begin Fetch: url1
Fetch Finished!
Begin Fetch: url2
Fetch Finished!
Begin Fetch: url3
Fetch Finished!
```

说明 AI 在串行调用 `web_fetch`。

解决：改用 `web_fetch_batch`。

### 18.5 需要完整页面内容

使用：

```json
{
  "url": "...",
  "mode": "full"
}
```

`mode=full` 会比 `summary` 慢，只在需要完整 Markdown 时使用。

### 18.6 修改后 Claude Code 或 Codex 里还是旧行为

重启对应 Agent。MCP 不会热更新。

### 18.7 npm install 后魔改被覆盖

如果在稳定部署目录重新执行 `npm install grok-search-mcp@latest`，`dist/providers/grok.js` 和 `dist/server.js` 可能被覆盖。

处理：

- 重新检查第 9、10 节魔改是否仍然存在。
- 必要时重新应用魔改。
- MCP 配置里的 `args` 仍然指向同一个稳定部署目录，不需要换路径。

### 18.8 Codex 看不到 MCP 工具

检查：

```bash
codex mcp list
codex mcp get grok-search
```

如果配置存在但当前会话看不到工具，重启 Codex。

### 18.9 Claude Code 看不到 MCP 工具

检查：

```bash
claude mcp get grok-search
```

如果配置存在但当前会话看不到工具，重启 Claude Code。

### 18.10 API key 错误

不要修改 `~/.config/grok-search/config.json`。

应更新：

```text
~/.claude.json
~/.codex/config.toml
```

对应 MCP env 里的：

```text
GROK_API_KEY
```

### 18.11 主会话调用 Grok MCP 被误报越狱拦截

**现象：** 在 Claude Code 主对话中直接调用 `web_search` 等 Grok MCP 工具，返回类似：

```text
此查询包含试图绕过 xAI 安全政策的越狱尝试……
```

即使查询内容完全正常（如"今天的 AI 新闻"）也会被拒绝。

**根因：** 当前会话上下文与其他用户对话混入，未做隔离，导致 Grok 过滤器误判。

**解决方法：** 用 `Agent` 工具开一个子代理，在干净的上下文里调用 Grok MCP：

```text
Agent prompt 示例：
请用 mcp__grok-search__web_search 工具搜索"<你的查询>"，返回结果摘要即可。
```

子代理从空白上下文启动，不携带主会话的混入历史，可绕过误报。

**注意：** 这是 troubleshooting 手段，不需要把所有搜索都默认走子代理。主会话大多数情况下直接调用即可，只在遭遇误报时才切换到子代理。

### 18.12 找不到 server.js

优先检查稳定部署目录。

Windows PowerShell：

```powershell
$ServerJs = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified\node_modules\grok-search-mcp\dist\server.js"
Test-Path $ServerJs
$ServerJs
```

Linux/macOS Bash：

```bash
SERVER_JS="${XDG_DATA_HOME:-$HOME/.local/share}/grok-search-mcp-unified/node_modules/grok-search-mcp/dist/server.js"
test -f "$SERVER_JS" && printf '%s\n' "$SERVER_JS"
```

如果文件不存在，回到第 5 节重新在稳定目录安装。

## 19. 实施后自检

部署 Agent 在提示用户重启前，应做一次本地静态自检。

Windows PowerShell：

```powershell
$DeployRoot = Join-Path $env:LOCALAPPDATA "grok-search-mcp-unified"
$ServerJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\server.js"
$GrokJs = Join-Path $DeployRoot "node_modules\grok-search-mcp\dist\providers\grok.js"

Select-String -Path $GrokJs -Pattern "GROK_REASONING_EFFORT","GROK_REQUEST_TIMEOUT_MS","response.text","AbortController","fetchBatch","wrapSingleResponseSearch","response.output_text.delta","extractFinalMessageText" -SimpleMatch
Select-String -Path $ServerJs -Pattern "web_fetch_batch","fetch_strategy","mode" -SimpleMatch
```

Linux/macOS Bash：

```bash
DEPLOY_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/grok-search-mcp-unified"
SERVER_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/server.js"
GROK_JS="$DEPLOY_ROOT/node_modules/grok-search-mcp/dist/providers/grok.js"

grep -nE 'GROK_REASONING_EFFORT|GROK_REQUEST_TIMEOUT_MS|response\.text|AbortController|fetchBatch|wrapSingleResponseSearch|response\.output_text\.delta|extractFinalMessageText' "$GROK_JS"
grep -nE 'web_fetch_batch|fetch_strategy|mode' "$SERVER_JS"
```

最低通过标准：

- `grok.js` 中能看到 `GROK_REASONING_EFFORT`、`GROK_REQUEST_TIMEOUT_MS`、`response.text()`、`AbortController`、`fetchBatch`、`wrapSingleResponseSearch`、`response.output_text.delta`、`extractFinalMessageText`。
- `server.js` 中能看到 `web_fetch_batch`、`fetch_strategy`、`mode`。
- Claude Code / Codex MCP 配置里的 `args` 指向稳定部署目录，不指向 npx 缓存。
- 输出给用户时不泄露 API key。

## 20. AI Agent 操作规则

1. 默认资料整理：调用 `web_search(fetch_strategy="single_response")`。
2. 多 URL 核实：调用 `web_fetch_batch(mode="summary", concurrency=3~5)`。
3. 单 URL 快速核实：调用 `web_fetch(mode="summary")`。
4. 完整网页内容：才调用 `web_fetch(mode="full")`。
5. 不要串行调用多个 `web_fetch`。
6. 不要默认使用 `web_search(fetch_strategy="multi_request_parallel")`。
7. 如果出现 Codex 120 秒工具调用超时，先查 `~/.config/grok-search/logs`，再检查 `GROK_REQUEST_TIMEOUT_MS` 和 `GROK_RETRY_COUNT` 是否生效。
8. 修改 MCP 代码或 MCP 配置后，提醒用户重启 Claude Code / Codex。
9. 部署时必须向用户索取真实 `GROK_API_KEY`，不要从其他配置文件复制 key。
10. 后续切换模型时，只改配置中的 `GROK_MODEL` 和 `~/.config/grok-search/config.json`，不要改魔改代码。
11. 主会话调用 Grok MCP 出现越狱误报拒绝时，改用 `Agent` 子代理发起独立会话调用，子代理上下文干净，可绕过误报。详见 18.11 节。

## 21. 部署完成后的用户提示

Agent 按本文档部署完成后，必须向用户输出下一步需要做什么。输出内容至少包含：

1. 已部署的 `server.js` 绝对路径。
2. 已写入的 MCP 客户端：Claude Code、Codex，或二者之一。
3. 当前 `GROK_MODEL`，当前 `GROK_REASONING_EFFORT=xhigh`。
4. 当前超时策略：`GROK_REQUEST_TIMEOUT_MS=85000`、`GROK_RETRY_COUNT=0`。
5. 提醒用户重启已配置的客户端，因为 MCP 配置和 Node 进程不会热更新。
6. 提醒用户重启后先运行 `get_config_info`，确认 `api_url`、`model`、API key 脱敏状态正常。
7. 提醒用户再运行一次 `web_search(fetch_strategy="single_response")` 做实测。
8. 如果出现 Codex 120 秒工具调用超时，先把 `GROK_REQUEST_TIMEOUT_MS` 从 `85000` 调到 `75000`，并保持 `GROK_RETRY_COUNT=0`。

推荐收尾话术：

```text
部署已完成。你接下来需要重启 Claude Code / Codex，让新的 MCP 配置和 server.js 魔改生效。

重启后先调用 grok-search.get_config_info，确认 api_url、model 和 API key 脱敏显示正常；再用 web_search(fetch_strategy="single_response") 做一次搜索测试。

当前使用 GROK_REASONING_EFFORT=xhigh，GROK_REQUEST_TIMEOUT_MS=85000，GROK_RETRY_COUNT=0。如果 Codex 仍出现 120 秒工具调用超时，优先把 GROK_REQUEST_TIMEOUT_MS 调到 75000。
```
