# 文档索引

项目安装、配置和当前运行流程见根目录 [README.md](../README.md)。

## 当前有效

- [HISTORY.md](HISTORY.md)：当前有效状态、按日期记录的实验过程与结论，以及仍然生效的文本策略、字幕时间轴策略、常见坑和参考来源。
- [translation-backend-architecture.md](translation-backend-architecture.md)：翻译侧 transport / profile / engine 三层契约与扩展指南。
- [audits/20260723_human-audit-page-core-v1.md](audits/20260723_human-audit-page-core-v1.md)：人工审计页共享 Core 的设计合同（`tools/audits/review_page_core.py` 仍在使用）。

## 归档

- [HISTORY-archive.md](HISTORY-archive.md)：已沉淀或被取代的历史记录。当前收纳 FusionVAD-JA、SpeechBoundary-JA、Qwen3-ASR 早期云端 SFT、1.7B Boundary / Scorer / CueQC / typed-span / pre-ASR 整条前置链、字幕时间轴早期实验与翻译后端早期重构。
- [archive/](archive/)：上述已退役链路的设计文档与审计报告。这些方案已被证伪或整体摘除，保留只为记录「哪些路已经走过」，不描述当前代码。
