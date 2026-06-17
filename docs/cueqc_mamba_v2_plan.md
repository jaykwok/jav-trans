# CueQC Mamba v2 — 历史废案

> 当前实现以 `docs/cueqc_mamba_v3_plan_extracted.md` 为准。v2 的 BGE-m3 文本 embedding、Boundary PTM/MFCC 音频序列、runtime 文本编码器和多模态方案已被 v3-Fusion 取代；不要按本文继续实现。

## 设计目标

CueQC Mamba 作为第二个 Mamba 模型，输入**音频时序特征 + ASR 文本嵌入 + chunk 元数据**，
输出 **display 二分类（keep/drop）**，自动决定字幕显示策略。模型像 boundary_refiner.pt
一样分发上传，runtime 全部模型（不保留规则回退，但保留异常 fallback）。

## 架构：音频 Mamba + 文本/structured 拼接

```
                    ┌─────────────────────────────────────┐
  音频 frame 序列   │  proj: Linear(ptm_dim+mfcc_dim→hidden)│
  [B, frames, 104] ─→│  Mamba2 (双向, forward+backward)     │──→ last hidden [B, hidden*2]
  (PTM 64 + MFCC 40)│  LayerNorm                           │
                    └─────────────────────────────────────┘
                                                          │ concat
  ASR 文本 BGE-m3 ─→ Linear(1024→text_proj_dim) ──────────┤
  [B, 1024]                                               │
                                                          │ concat
  structured 17 维 ─→ （直接拼接）────────────────────────┤
  [B, 17]                                                 │
                                                          ↓
                                              MLP(hidden*2+text_proj+17 → hidden → 2)
                                                          ↓
                                              display logits [B, 2] (keep/drop)
```

### 设计要点

1. **音频走 Mamba2**：复用 `TinyMamba2BoundaryBackbone`（双向 Mamba2，和 boundary refiner 同一个 backbone），
   输入 `[B, frames, ptm_dim+mfcc_dim]`，取最后一帧的隐状态 `[B, hidden*2]`。
   变长序列用 attention_mask 处理（padding 到 batch 内最大 frame 数）。

2. **文本走线性投影**：BGE-m3 sentence embedding（1024 维）→ `Linear(1024→text_proj_dim=128)`。
   用 `enrich_embeddings.py` 已有的 BGE-m3 提取（`CUEQC_TEXT_EMBED_MODEL`）。

3. **structured 直接拼接**：17 维 `numeric_feature_vector`，不额外投影（信息密度高、维度低）。

4. **融合层**：`concat([audio_hidden, text_proj, structured])` → `Linear(hidden*2+128+17 → hidden) → ReLU → Linear(hidden→2)`。

5. **变长处理**：每个 chunk 的 frame 数 = `ceil(duration_s / 0.02)`，batch 内 padding 到最大值 + attention_mask。

## 音频特征：复用 boundary 的 PTM + MFCC

### 关键复用（零成本）

| 阶段 | 来源 | 说明 |
|---|---|---|
| **训练** | `load_audio_16k_mono(wav_path)` → `audio[start*16000:end*16000]` → `extract_mfcc` + PTM | 从 baseline wav 切片提取 |
| **Runtime** | `FrameSequenceFeatureProvider`（boundary backend 已产出全片 PTM+MFCC frame）→ 按 chunk `[start,end]` 切片 | **不需要重新提取**，boundary 已经算好了 |

特征一致性：训练和 runtime 用同样的 `extract_mfcc(n_mfcc=40, n_fft=400, frame_hop_s=0.02)` + PTM（max 64 维），
同样的 `frame_hop_s=0.02`。训练时从 wav 提取，runtime 时从 provider 切片——同一份帧数据。

### 新增：`FrameSequenceFeatureProvider.frames_for_window(start_s, end_s)`

现有 `features_for_boundary` 是按 boundary window（left/gap/right）切并做 mean+std 聚合。
CueQC 需要的是**原始 frame 序列**（不聚合），所以给 provider 加一个方法：
```python
def frames_for_window(self, start_s, end_s) -> tuple[np.ndarray, np.ndarray]:
    """返回 [frames, ptm_dim] 和 [frames, mfcc_dim] 的原始帧序列。"""
    lower = max(0, round(start_s / self.frame_hop_s))
    upper = min(self._ptm_used.shape[0], round(end_s / self.frame_hop_s))
    return self._ptm_used[lower:upper], self._mfcc_used[lower:upper]
```

## 文本特征：BGE-m3 sentence embedding

### 训练时提取

`enrich_embeddings.py` 已有 BGE-m3 提取（`sentence-transformers`, `BAAI/bge-m3`，1024 维）。
流程：
1. 对 cold-start 训练数据跑 `enrich_embeddings.py`（先关联 audio path）→ 每 row 加 `embeddings.text`
2. 训练时从 row 读 `embeddings.text`（1024 维）

### Runtime 时提取

Runtime 时 CueQC 拿到的 candidate 已有 `text`/`raw_text`（ASR 输出）。
需要一个轻量的 BGE 编码器在 runtime 对每个 chunk 的文本编码。
但这有开销——每条音频的几百个 chunk 都要跑一次 BGE。

**方案**：runtime 用 BGE-m3 编码（`sentence-transformers`，和 boundary 的 Qwen3 一样在 GPU 上批量跑），
结果缓存在 candidate 级别（同一音频内重复文本不重复编码）。

## 改动文件清单

### 1. `src/boundary/sequence_features.py` — 加 `frames_for_window` 方法
- `FrameSequenceFeatureProvider` 加 `frames_for_window(start_s, end_s)` 返回原始 frame 序列

### 2. `src/asr/cueqc.py` — 加音频/文本特征提取辅助
- `extract_chunk_audio_frames(candidate, *, audio_provider=None, audio_path=None)` 
  - runtime：从 audio_provider 切片
  - 训练：从 audio_path + start/end 切片提取 MFCC+PTM
- `extract_text_embedding(text, *, model=None)` — BGE-m3 编码（带缓存）

### 3. `src/asr/cueqc_model.py` — 新建，多模态模型定义
```python
class CueQCMultimodalModel(nn.Module):
    def __init__(self, *, ptm_dim=64, mfcc_dim=40, text_dim=1024, structured_dim=17,
                 hidden_size=128, num_layers=2, ...):
        # 音频 backbone: TinyMamba2BoundaryBackbone(input_dim=ptm_dim+mfcc_dim, bidirectional=True)
        # 文本投影: Linear(text_dim, text_proj_dim)
        # 融合 MLP: Linear(hidden*2 + text_proj_dim + structured_dim, hidden) → ReLU → Linear(hidden, 2)
    def forward(self, audio_frames, audio_mask, text_emb, structured):
        # audio_frames: [B, T, ptm_dim+mfcc_dim], audio_mask: [B, T]
        # 返回 display logits [B, 2]
```

### 4. `tools/asr/cueqc/extract_features.py` — 新建，离线特征提取
- 读 `cueqc_train.jsonl` + baseline wav → 提取每 chunk 的音频 frame 序列 + 文本 BGE 嵌入
- 输出 `cueqc_train_features.pt`（tensor: audio_frames padded + mask + text_emb + structured + label）
- 关联 wav：`audio.path` 重定向到 baseline root

### 5. `tools/asr/cueqc/train_mamba.py` — 重写为多模态训练
- 读 `cueqc_train_features.pt`
- 构建 `CueQCMultimodalModel`
- display 单头 CrossEntropy loss
- 变长音频 padding + collate_fn
- checkpoint schema `cueqc_mamba_checkpoint_v2`（存 model_config 含 modality dims + feature_mean/std + 音频/text 配置）
- train/val accuracy 报告

### 6. `src/asr/cueqc_refiner.py` — 新建，runtime adapter + loader
```python
class CueQCRefiner:
    def decide(self, candidates, *, audio_provider, text_encoder) -> list[dict]:
        # 对每个 candidate: 提取音频 frames（从 provider 切片）+ 文本 embedding + structured
        # batch forward → argmax → {display_hint, confidence, mode:"model", ...}
```

### 7. `src/asr/pipeline.py` — `_run_cueqc_shadow` 接入模型
- 当 `CUEQC_MODEL_PATH` 存在 → 加载 `CueQCRefiner` → 传入 `audio_provider`（boundary 已产出）→ `decide(candidates)`
- 传 text_encoder（BGE-m3，模块级缓存）
- 异常 → fallback 全部 align

### 8. `src/asr/cueqc.py` runtime_signature + `src/core/config.py` 默认路径
- `CUEQC_MODEL_PATH` 默认 `src/asr/checkpoints/cueqc_mamba.pt`
- `CUEQC_MODEL_VERSION` → `cueqc_mamba_v2`
- `CUEQC_TEXT_EMBED_MODEL` 默认 `BAAI/bge-m3`

## 训练流程

```bash
# 1. 关联音频 + 提取特征（MFCC+PTM frames + BGE text embedding）
uv run python -m tools.asr.cueqc.extract_features \
  --train agents/temp/20260616_clueqc-cold-start-train/cueqc_train.jsonl \
  --audio-root agents/temp/20260615_094437_b5/agents/temp/speech-boundary-ja/20260615_094437_o10 \
  --output agents/temp/20260616_clueqc-mamba-v2-train/cueqc_train_features.pt

# 2. 训练
uv run python -m tools.asr.cueqc.train_mamba \
  --features agents/temp/20260616_clueqc-mamba-v2-train/cueqc_train_features.pt \
  --output-dir agents/temp/20260616_clueqc-mamba-v2-train \
  --max-steps 300 --val-ratio 0.15

# 3. 分发
cp agents/temp/20260616_clueqc-mamba-v2-train/cueqc_mamba.pt src/asr/checkpoints/cueqc_mamba.pt
```

## 数据约束

- 300 条 cold-start 种子，display keep(133)/drop(167)
- 音频：从 baseline 10 部影片 wav 切片（chunk 时长 0.4-3s → frames 20-150）
- 文本：BGE-m3 1024 维（`enrich_embeddings.py` 已验证可用）
- structured：17 维

## 验证
1. py_compile 全部新文件
2. 特征提取跑通（300 条，检查 frame 数 + embedding 维度）
3. 训练 val accuracy（多模态应显著优于纯 structured，目标 > 0.8）
4. runtime 冒烟：pipeline 设 CUEQC_MODEL_PATH → decision_by_chunk 有值
5. fallback 冒烟：MODEL_PATH 不存在 → 全部 align
6. HISTORY 记录

## 不改动
- boundary backbone（`backbones.py`）
- boundary refiner / pipeline boundary 路径
- cluster_candidates.py / torque.py / compile_training_set.py
- 审计页生成器

## 后续（不在本次）
- 阶段 1：Mamba 隐状态聚类（暴露 audio backbone hidden state 做时序聚类）
- 阶段 2：自训练（模型出伪标签→重训）
- Boundary 反哺（display=drop → boundary preference）
- compact/review 头（如需）
