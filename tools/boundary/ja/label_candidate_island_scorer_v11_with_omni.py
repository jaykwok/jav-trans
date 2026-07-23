#!/usr/bin/env python3
"""Create model-independent Scorer v11 candidate-island teacher preaudits.

This is deliberately a preaudit, never a canonical-truth compiler.  The
teacher labels the continuous dialogue/candidate envelope that Scorer should
preserve.  It must not split sentences, dialogue turns, or ASR units; Proposal
and Split own that later decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from tools.asr.cueqc.label_pre_asr_with_omni import (
    DEFAULT_API_KEY_ENV_CANDIDATES,
    DEFAULT_BASE_URL_ENV_CANDIDATES,
    DEFAULT_ENV_FILE,
    call_omni,
    first_env_value,
    load_env_file,
    normalize_openai_compat_base_url,
)


FRAME_HOP_S = 0.02
SCHEMA = "candidate_island_scorer_v11_omni_preaudit_v2"
SUMMARY_SCHEMA = "candidate_island_scorer_v11_omni_preaudit_summary_v2"
DEFAULT_PROMPT_PROFILE = "dialogue-islands-v5"
SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v1"
SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v2-simple"
GREENLIGHT_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v3-greenlight"
FUNNEL_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v4-funnel"
ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v5-assertive"
BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-complement-v6-balanced-v12-teacher"
CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE = "safe-outside-custom-file"
SAFE_OUTSIDE_PROMPT_PROFILES = (
    SAFE_OUTSIDE_PROMPT_PROFILE,
    SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE,
    GREENLIGHT_SAFE_OUTSIDE_PROMPT_PROFILE,
    FUNNEL_SAFE_OUTSIDE_PROMPT_PROFILE,
    ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE,
    BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE,
    CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE,
)
PROMPT_VERSION = "candidate_island_scorer_v11_omni_preaudit_dialogue_islands_v5"
SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v1"
)
SIMPLE_SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v2_simple"
)
GREENLIGHT_SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v3_greenlight"
)
FUNNEL_SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v4_funnel"
)
ASSERTIVE_SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v5_assertive"
)
BALANCED_V12_SAFE_OUTSIDE_PROMPT_VERSION = (
    "candidate_island_scorer_v11_omni_preaudit_safe_outside_complement_v6_balanced_v12_teacher"
)

SYSTEM_PROMPT = """你是 1.7B Scorer v11 的候选岛预审 teacher。你的唯一职责是标出必须先保留给后续 Proposal / Split / CueQC 的连续候选对话岛。你不是 Split，不按句子、说话人、标点或语义单元切分；你也不是 CueQC，不做最终 keep/drop。

必须按以下优先顺序判断：
1. 先寻找明确或很可能含词语、音节、耳语、口吃、残缺发音、句尾或对白的波形，把它们作为 inside_candidate 锚点。不要以 ASR 能否转录作为判据。
2. 再围绕这些锚点形成连续候选岛。同一轮连续对话、几乎无安全间隔的相邻发言，以及对白内部或边缘的停顿、尾音、短呼吸、呻吟或动作声，应保持在同一个岛中，保证完整波形交给下游。句子和事件切分属于 Proposal + Split。
3. 明确不含词语且能够独立于对白删除的纯呻吟、喘息、呼吸、哭声、亲吻声、动作声、impact、音乐、静音或环境声属于 outside_candidate。即使它持续很久、强度很高、有人互动或与对白处于同一场景，也不能仅因此成为 inside_candidate。若整条 source 都是明确的纯非语义声音，必须允许 islands=[]。
4. 非语义声音只有在夹在同一轮对话内部、紧贴对白边缘，且移除会截断尾音或破坏连续对话波形时，才随该对话岛保留。纯非语义声音本身不能桥接相距较远的两轮对白。
5. 若局部听起来可能是词语、也可能只是呻吟或噪声，优先标为 unsure；不要为了高召回直接扩大成 inside_candidate。unsure 是标注不确定性，之后会映射为 ignore=-100。

边界合同：
- 不使用固定时长、静音阈值、hysteresis、ASR 文本、duration-only 规则或其他启发式。
- 同一场景、同一说话人、持续互动或声音连续，本身都不是合并理由。
- islands 与 unsure_spans 必须各自按时间排序、互不重叠，并且两组之间也不得重叠；它们之外的完整差集就是 outside_candidate。
- 输出当前 0-based 完整 source 坐标，单位为秒，不添加前后文，不使用原视频时间轴。

判例：
- 对白1 + 对白内部短呼吸/呻吟 + 对白2，且属于同一轮连续对话：输出一个完整 island。
- 全段只有明确纯呻吟/喘息，没有词语：islands=[]，不要因为声音连续而整段保留。
- 某段可能是词语，也可能只是呻吟：该局部输出 unsure_span。
- 对白 + 独立纯非语义活动 + 后续另一轮对白：输出两个 islands，中间差集为 outside_candidate。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "islands":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"连续候选对话岛的简短理由"}],
  "unsure_spans":[{"start_s":0.0,"end_s":1.0,"reason":"无法确认是否含词语的局部"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的高精度 outside_candidate 预审 teacher。音频主要来自日语 JAV / Galgame 场景，但场景、亲密互动或声音类型本身不能决定标签。你的唯一任务是找出在任何下游模型运行前即可整段安全删除的稀疏区间；不要标注 inside_candidate，也不要尝试切句。

一个区间只有同时满足以下全部条件，才允许输出为 safe_outside_span：
1. 整个区间确认不含任何可能的日语词语、应答、音节、耳语、口吃、残缺发音、句尾或对白。背景中的人声只要可能含词，也不是安全 outside。
2. 删除整个区间不会截断前后词尾，不会切开同一轮连续对白，也不会把紧邻对白拆成碎片。
3. 起止点精确排除相邻的可能语音；宁可少报，不可把可能语音一起框入。

必须特别注意：呻吟、喘息和呼吸中可能夹有有意义的发声。中文近似“啊、嗯、哼、哈、诶”，以及日语「あ、あっ、うん、ん、ふん、え、はぁ」等，可能是应答、感叹、音节、词尾或语用表达，不能因为短、像呻吟或 ASR 未识别就删除。只要存在这种可能，就完全不要输出该局部，由补集保守保留。

可以考虑但绝不能按类别自动输出的声音包括：确定非词化的纯呻吟/喘息/呼吸/哭声、亲吻或舔舐声、身体/床/衣物/器械动作与 impact、静音、BGM/音乐、房间底噪、风扇/空调、电流嗡声、嘶声/静电、风雨、交通、机械声、麦克风摩擦及其他环境背景噪声。它们仍必须满足“无任何可能词语”和“可独立删除”两个条件。夹在同一对白包络内、紧贴词尾或边界不确定的声音不要输出。

不确定策略：不要输出 unsure_spans；只需省略不确定区间。省略的完整补集只是 provisional keep 候选，不代表已经确认的 canonical inside truth。允许 safe_outside_spans=[]；只有整条 source 都确定无任何可能词语且可整体安全删除时，才允许输出覆盖全段的一个区间。

边界合同：
- 不使用 ASR 文本、固定时长、静音阈值、hysteresis、duration-only 规则或其他启发式。
- safe_outside_spans 必须按时间排序、互不重叠，使用当前 0-based 完整 source 坐标，单位为秒；不添加上下文，不使用原视频时间轴。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认无词语且可独立删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的 Outside Candidate（可剪裁非语言区间）预审 Teacher 模型。
你的目标是：识别音频中不含任何有效对白与语言信息的“纯背景/纯非语言区间”，以便后续安全删除。

在 JAV / Galgame 场景下，请严格遵循以下【判定逻辑】与【边界规则】：

一、允许标记为 outside 的区间（必须满足高置信度）：
1. 纯环境与物理音：无对白的静音、底噪、风扇/空调声、电流声、机械声、摩擦声、BGM/背景音乐。
2. 独立且连续的非语言生理音：纯粹的呼吸声、连续喘息、哭声、亲吻声、吞咽声等（前提：该区间内没有任何清晰的日语词语、对话或明确的互动应答）。
注：必须是与对白有明显间隔的独立片段，或持续性无语义音段。

二、必须保留（绝对不能标为 outside，视为 Inside/Keep）：
1. 所有清晰或模糊的日语对白、词语、耳语、口吃、句尾。
2. 具有互动/应答属性的短感叹词：如「あ、あっ、うん、ん、ふん、え、はぁ」等（即使 ASR 识别不出，只要发生在对话上下文、或像是在对答，就必须保留）。
3. 人声与对白紧密相连的边缘：无法明确区分人声和背景音的重叠区域。

三、边界切割规则（防止切词）：
- 安全缓冲区：在遇到真实人声/对白的前后，必须留出至少约 0.2～0.3 秒的缓冲区。宁可把 outside 范围缩小，也绝不能贴着人声边缘切割。
- 如果一段非语言声音（如喘息）与后续的对白没有任何停顿，请放弃标记该段的前半部分，仅标记远离人声的纯净段落。

四、输出要求：
- 仅输出确认可以删除的 outside_candidate 起止时间段。
- 难以抉择的模糊片段直接忽略（不输出）。
- 保持高准确率的同时，积极找出确凿的纯静音、纯噪音段落以及远离对白的纯喘息/环境音段落。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认可安全删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

GREENLIGHT_SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的高精度 outside_candidate 预审 teacher。音频主要来自日语 JAV / Galgame 场景。
你的任务是找出在任何下游模型运行前，可以安全删除的“非对白/非语言”稀疏区间（标记为 outside）。不需要标注 inside_candidate，不切句。

【Outside 判定核心原则】
请区分“语言性发音”与“纯生理性/物理性声音”。只有确定属于后者且能独立删除的区间，才能标记为 outside。

✅ 必须标记为 Outside 的情况（绿灯）：
1. 纯物理/环境音：静音、BGM、环境底噪、风扇、空调、电流声、交通、机械、衣物摩擦、动作拍打声 (impact)。
2. 纯生理性非语言声音：确定不含语义的持续喘息、深呼吸、哭泣、亲吻声、水声。
3. 纯生理性呻吟：虽然 ASR 可能将其识别为「あ、あっ、はぁ、ん」等，但如果它们是连续、有节奏的生理反应（如长段的高潮/喘息），且前后没有连接任何具体的日语词汇或对白，可以判定为 outside。

❌ 绝对禁止标记为 Outside 的情况（红灯）：
1. 任何包含明确日语词汇、短语、句子的片段（如 やめて, 気持ちいい 等）。
2. 具有“交际/应答”属性的叹词：如果「うん、え、あ」用于回答问题、表达惊讶或作为附和（相槌），即使很短也必须保留（不能标 outside）。
3. 夹杂在连续对白中间的极短停顿（删除会导致前后语意碎片化或截断词尾）。

【边界处理规范】
- 完整性：outside 区间的起止点应尽量包裹完整的非语音事件，不要切碎同一轮连续对白。
- 保守但行动：不确定的模糊区间直接忽略，不输出 unsure。但对于特征明显的长段纯 BGM 或纯生理喘息，请自信标记。未标记部分仅作为 provisional keep。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认可安全删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

FUNNEL_SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的高精度 outside_candidate 预审 teacher（针对日语 JAV/Galgame 音频）。
你的任务是筛选出绝对可以安全删除的非对白片段（标记为 outside）。

请根据以下【判定漏斗】对每个片段进行判断，只有通过所有检查的才能标为 outside：

1. 语义检查：这段声音是否包含任何日语词汇、对话或残缺的口吃发音？
   - 是 -> 忽略（不标记）。
   - 否 -> 进入下一步。
2. 应答检查：声音中的「あ、ん、え」等，是在进行交流附和（相槌/回答），还是纯粹的生理性呻吟/喘息？
   - 是交流附和 -> 忽略（不标记）。
   - 是纯生理呻吟/喘息/亲吻/哭泣 -> 进入下一步。
3. 物理检查：这段声音是否完全是静音、音乐、底噪、摩擦声或动作音效？
   - 是 -> 进入下一步。
4. 切割安全检查：将这段删除后，是否会切断前后对话的词尾，或导致正常句子破碎？
   - 是 -> 忽略（不标记）。
   - 否 -> 确定标记为 outside。

【约束】
- 只要确信是“无交际意图的纯生理声音”或“纯环境音/噪音”，就应该果断标记 outside，不要因为 ASR 输出了无意义的拟声词（如「あっ」）就退缩。
- 遇到无法判断的区间直接跳过，绝对不要输出 unsure。
- 你的标记将作为删除依据，起止点需尽量精准。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认可安全删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v11 的高精度 outside_candidate 预审 teacher。音频主要来自日语 JAV / Galgame 场景。
【核心任务】找出音频中所有【无语言语义价值】的片段（标注为 outside_candidate），以便后续整段安全删除。
在本项目中，【非语义的呻吟、喘息、呼吸】属于重点清理的目标，必须积极标记为 outside。

【准许标记为 outside 的范围（满足其一即可）】
1. 非语义发声（重点标记）：
   - 连续或独立的呻吟声（Moan）、喘息声（Panting/Gasping）、重呼吸声（Breathing）。
   - 生理与动作音：亲吻声、吞咽声、水声、摩擦声、哭泣抽泣声、拍打/撞击声。
   判别标准：只要该发声仅为“生理/情绪本能发音”，且不承载具体的文本对白或语言应答，一律标记为 outside。

2. 非语音环境音：
   - 纯静音、环境底噪（电流声、风扇、空调、环境杂音）、纯背景音乐（BGM）。

【绝对禁止标记为 outside（必须保留的语音）】
1. 具有语言学对白/应答含义的声音：
   - 明确的日语单词、句子、句子残缺发音。
   - 具有明确“对话/应答功能”的短音（例如：表示确认/答应的「うん」、表示疑问/惊讶的「え？」、表示赞同的「はい」等）。
2. 切口安全界限：
   - 标记起止点必须与真正的对白/词头词尾保持至少 50ms 的安全距离，绝不能截断正常对白的边缘。

【关键判定逻辑】
- 不要因为呻吟声中合成了类似「啊、哈、嗯」的音素就把呻吟误判为对白。
- 区分核心：是“单纯的生理呻吟/喘息/发泄（标记 outside）”，还是“带有语义的对话/应答（禁止标记）”。
- 遇到明确的无语义呻吟段、喘息段、静音段，请大胆标记；仅在无法区分是否为对白时才省略。

未标记部分仅作为 provisional keep。只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认可安全删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""

BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT = """你是 1.7B Scorer v12 的 outside_candidate 预审 teacher。

音频主要来自日语 JAV、Galgame 或类似场景，但场景、亲密互动、说话人身份以及声音类型本身都不能直接决定标签。

你的任务是找出在任何下游模型运行前，具有较高把握可以整段删除的非语言区间，并将其标记为 outside_candidate。

不要标注 inside_candidate，不要切分或转写对白。

核心判断原则

当一个区间整体上明显属于非语言声音，并且没有可辨认的词、音节组合或交流性应答时，可以标记为 outside_candidate。

不要求证明该区间“绝对不可能”包含语言。应根据可听证据判断其主要性质，而不是因为存在极低概率的语言可能性就全部保留。

优先保护真实或疑似语言，但不要把所有带有人声色彩的声音都视为语言。

应当保留的内容

以下任一情况存在时，不要标记为 outside_candidate：

能听出完整或残缺的日语词语、短语、对白或句尾。
能听出具有语言结构的连续音节，即使 ASR 无法识别。
明显属于应答、呼唤、否定、肯定、疑问或其他交流行为。
出现口吃、重复、吞音、耳语或不完整发音，但仍可能属于某个词。
删除后会明显截断前后语音，或切开同一轮连续对白。
区间过短，无法可靠地区分语言音节和非语言声音。

「あ」「あっ」「うん」「ん」「ふん」「え」「はぁ」以及“啊、嗯、哼、哈、诶”等短音，不应仅凭文字形式直接保留或删除。

需要结合实际声音判断：

如果具有明确的应答、感叹、呼唤或语言音节功能，应保留。
如果只是连续呻吟、呼吸释放、喘息或无交流性的发声，可以标记为 outside_candidate。
如果无法可靠判断其是否承担语言功能，则省略该区间。

可以标记的 outside_candidate

在没有可辨语言结构的前提下，以下内容可以标记：

非词化呻吟、持续拖长音或重复的无语义发声。
喘息、吸气、呼气、急促呼吸和叹气。
哭声、抽泣、笑声以及其他非词化情绪声音。
亲吻声、吞咽声、口腔声和唾液声。
动作声、撞击声、拍打声、摩擦声和床体或衣物声音。
静音、长停顿和仅含底噪的区间。
音乐、环境声、风扇、空调、电流、交通或机械噪声。
不构成词语或交流行为的孤立人声。

声音是由人发出的，并不代表它一定是语言。只要缺乏可辨识的词汇、音节结构和交流功能，非词化人声也可以标记。

边界规则

outside_candidate 的边界应避开清晰或疑似语言，但不需要为了排除极弱、不可辨认的声音而无限收缩。

设置起止点时：

保留语音开始前和结束后的少量安全余量。
不要截断清晰的辅音、元音、词尾或耳语。
可以包含与目标非语言声音连续相连的轻微呼吸、底噪或动作声。
一个较长非语言区间中如果只出现极弱、不可辨认且没有语言结构的瞬时人声，不必因此放弃整个区间。
如果中间出现明确或较强的疑似词语，应将区间拆开或不标记该部分。

决策优先级

按以下顺序判断：

是否存在可辨认的词、对白或语言结构。
是否存在明确的交流功能。
区间整体是否主要由非语言声音构成。
删除后是否会破坏相邻语音的完整性。

只有前两项均为否，且后两项均为是时，标记为 outside_candidate。

输出策略

只输出具有合理把握的 outside_candidate。
不确定区间直接省略，不输出 unsure。
未标记内容仅代表 provisional keep，不代表 canonical inside truth。
目标是在保护真实语言的同时，积极找出明显的非语言区间。不要因为声音具有音高、情绪、声带振动或类似人声，就自动拒绝标记。

只输出一个 JSON 对象，不要 Markdown：
{
  "source_id":"...",
  "safe_outside_spans":[{"start_s":0.0,"end_s":1.0,"confidence":0.0,"reason":"确认可安全删除的简短理由"}],
  "overall_confidence":0.0,
  "overall_reason":"简短整体理由"
}
"""


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        if os.path.exists(raw):
            os.unlink(raw)


def _resume_index(
    path: Path,
    *,
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_id"]): row
        for row in _rows(path)
        if row.get("schema") == SCHEMA
        and row.get("model") == model
        and row.get("prompt_version") == prompt_version
    }


def _resolve_audio(value: str, *, manifest: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [manifest.parent / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(value)


def _prompt(
    row: Mapping[str, Any],
    *,
    feedback: str = "",
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
) -> str:
    if prompt_profile == DEFAULT_PROMPT_PROFILE:
        payload = {
            "source_id": str(row["source_id"]),
            "duration_s": float(row["duration_s"]),
            "task": "mark continuous candidate dialogue islands for Scorer v11",
            "decision_order": [
                "find definite or probable lexical/dialogue anchors",
                "preserve the continuous waveform envelope of the same dialogue round",
                "leave independently removable definite nonlexical sound in the outside complement",
                "mark locally ambiguous possible words as unsure rather than defaulting to inside",
            ],
            "do_not_split": [
                "the same continuous dialogue round",
                "adjacent dialogue turns with almost no safely removable interval",
                "intra-dialogue pauses, pronunciation tails, breaths, or action sounds",
            ],
            "outside_candidate": "the complement containing definite independent nonlexical sound; a continuous pure moan/pant/breath scene may be entirely outside",
            "output_units": "continuous dialogue-candidate envelopes, never individual sentences or semantic units",
            "must_split": "independent definite nonlexical activity between separate dialogue rounds ends the prior island even when the scene and interaction continue",
            "nonsemantic_vocal_policy": "keep nonlexical sound only when attached to or enclosed by the same dialogue envelope; otherwise outside; possible lexical ambiguity goes to unsure",
            "anti_overmerge": "never return 0..duration merely because vocal activity, intimacy, emotion, or interaction is continuous; if there is no definite or probable dialogue and no ambiguity, return islands=[]",
            "range_contract": "islands and unsure_spans are sorted and mutually exclusive; their omitted complement is outside_candidate",
        }
    elif prompt_profile == SAFE_OUTSIDE_PROMPT_PROFILE:
        payload = {
            "source_id": str(row["source_id"]),
            "duration_s": float(row["duration_s"]),
            "task": "return only high-precision independently removable nonlexical spans",
            "required_conditions": [
                "the whole span contains no possible Japanese word, response, syllable, whisper, stutter, partial pronunciation, pronunciation tail, or dialogue",
                "removing the whole span does not clip or fragment the same continuous dialogue round",
                "the coordinates exclude every adjacent possibly lexical frame",
            ],
            "nonlexical_examples_to_consider": [
                "definitely nonlexical moan, pant, breath, cry, kiss, lick, body/bed/clothing/action impact",
                "silence, BGM/music, room tone, HVAC/fan, electrical hum, hiss/static, wind/rain, traffic, machinery, microphone rub, and other environmental background noise",
            ],
            "lexical_ambiguity_warning": "vocalizations resembling 啊/嗯/哼/哈/诶 or Japanese あ/あっ/うん/ん/ふん/え/はぁ may be responses, interjections, syllables, or pronunciation tails; omit them whenever that is possible",
            "background_voice_policy": "any background human voice that may contain a word is not safe outside",
            "uncertainty_policy": "omit the interval; the omitted complement is provisional keep, not confirmed canonical inside truth",
            "range_contract": "safe_outside_spans are sparse, sorted, non-overlapping, and use the current 0-based full-source timeline",
        }
    elif prompt_profile in SAFE_OUTSIDE_PROMPT_PROFILES:
        payload = {
            "source_id": str(row["source_id"]),
            "duration_s": float(row["duration_s"]),
            "coordinate_system": "0-based current full-source timeline in seconds",
        }
    else:
        raise ValueError(f"unknown Scorer v11 teacher prompt profile: {prompt_profile}")
    if feedback:
        payload["previous_validation_error"] = feedback
    return json.dumps(payload, ensure_ascii=False)


def _number(value: Any, *, name: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def _spans(parsed: Mapping[str, Any], *, duration_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    islands: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(parsed.get("islands") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"island {index} is not an object")
        start = float(raw.get("start_s")); end = float(raw.get("end_s"))
        if not 0.0 <= start < end <= duration_s or start < previous_end:
            raise ValueError(
                f"island {index} has invalid local-source coordinates: "
                f"start_s={start}, end_s={end}, previous_end_s={previous_end}, "
                f"required_range=0..{duration_s}; use this 0-based audio clip timeline, "
                "never timestamps from the original video"
            )
        previous_end = end
        islands.append({"label": "inside_candidate", "start_s": start, "end_s": end, "start_frame": round(start / FRAME_HOP_S), "end_frame": round(end / FRAME_HOP_S), "confidence": _number(raw.get("confidence", 0.0), name="island confidence"), "reason": str(raw.get("reason") or "")})
    unsure: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, raw in enumerate(parsed.get("unsure_spans") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"unsure span {index} is not an object")
        start = float(raw.get("start_s")); end = float(raw.get("end_s"))
        if not 0.0 <= start < end <= duration_s or start < previous_end:
            raise ValueError(
                f"unsure span {index} has invalid local-source coordinates: "
                f"start_s={start}, end_s={end}, previous_end_s={previous_end}, "
                f"required_range=0..{duration_s}; use this 0-based audio clip timeline, "
                "never timestamps from the original video"
            )
        previous_end = end
        unsure.append({"label": "unsure", "start_s": start, "end_s": end, "start_frame": round(start / FRAME_HOP_S), "end_frame": round(end / FRAME_HOP_S), "reason": str(raw.get("reason") or "")})
    classified = sorted(
        (
            (float(span["start_s"]), float(span["end_s"]), str(span["label"]))
            for span in (*islands, *unsure)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for previous, current in zip(classified, classified[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "Scorer v11 teacher islands and unsure_spans must be mutually "
                f"exclusive: {previous[2]} {previous[0]}..{previous[1]} overlaps "
                f"{current[2]} {current[0]}..{current[1]}"
            )
    return islands, unsure


def _safe_outside_complement(
    parsed: Mapping[str, Any],
    *,
    duration_s: float,
    frame_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if frame_count <= 0:
        raise ValueError("outside-first teacher requires a positive frame_count")
    safe_outside: list[dict[str, Any]] = []
    previous_end_frame = 0
    for index, raw in enumerate(parsed.get("safe_outside_spans") or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"safe outside span {index} is not an object")
        start = float(raw.get("start_s"))
        end = float(raw.get("end_s"))
        if not 0.0 <= start < end <= duration_s:
            raise ValueError(
                f"safe outside span {index} has invalid local-source coordinates: "
                f"start_s={start}, end_s={end}, required_range=0..{duration_s}; "
                "use this 0-based audio clip timeline, never timestamps from the original video"
            )
        start_frame = 0 if start <= 0.0 else round(start / FRAME_HOP_S)
        end_frame = (
            frame_count
            if duration_s - end < FRAME_HOP_S
            else round(end / FRAME_HOP_S)
        )
        if (
            start_frame < previous_end_frame
            or start_frame < 0
            or end_frame > frame_count
            or end_frame <= start_frame
        ):
            raise ValueError(
                f"safe outside span {index} has invalid/overlapping frame coordinates: "
                f"start_frame={start_frame}, end_frame={end_frame}, "
                f"previous_end_frame={previous_end_frame}, frame_count={frame_count}"
            )
        previous_end_frame = end_frame
        safe_outside.append(
            {
                "label": "outside_candidate",
                "start_s": round(start_frame * FRAME_HOP_S, 6),
                "end_s": round(end_frame * FRAME_HOP_S, 6),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "confidence": _number(
                    raw.get("confidence", 0.0),
                    name="safe outside confidence",
                ),
                "reason": str(raw.get("reason") or ""),
            }
        )

    islands: list[dict[str, Any]] = []
    cursor = 0
    for span in safe_outside:
        start_frame = int(span["start_frame"])
        if cursor < start_frame:
            islands.append(
                {
                    "label": "inside_candidate",
                    "start_s": round(cursor * FRAME_HOP_S, 6),
                    "end_s": round(start_frame * FRAME_HOP_S, 6),
                    "start_frame": cursor,
                    "end_frame": start_frame,
                    "confidence": 0.0,
                    "reason": "outside-first provisional keep complement; not confirmed canonical inside truth",
                }
            )
        cursor = int(span["end_frame"])
    if cursor < frame_count:
        islands.append(
            {
                "label": "inside_candidate",
                "start_s": round(cursor * FRAME_HOP_S, 6),
                "end_s": round(frame_count * FRAME_HOP_S, 6),
                "start_frame": cursor,
                "end_frame": frame_count,
                "confidence": 0.0,
                "reason": "outside-first provisional keep complement; not confirmed canonical inside truth",
            }
        )
    return islands, [], safe_outside


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = str(args.env_file)
    env_file = Path.home() / ".config" / "omni" / profile
    load_env_file(env_file)
    _model_env, model = first_env_value(tuple(args.model_env.split(",")))
    model = args.model or model
    _key_env, api_key = first_env_value(tuple(args.api_key_env.split(",")))
    _url_env, raw_base_url = first_env_value(tuple(args.base_url_env.split(",")))
    base_url = normalize_openai_compat_base_url(raw_base_url)
    if not model or not api_key:
        raise RuntimeError("Omni model and API key are required")
    prompt_profile = str(args.prompt_profile)
    prompt_source_file = ""
    prompt_source_sha256 = ""
    if prompt_profile == DEFAULT_PROMPT_PROFILE:
        prompt_version = PROMPT_VERSION
        system_prompt = SYSTEM_PROMPT
    elif prompt_profile == SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == SIMPLE_SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = SIMPLE_SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = SIMPLE_SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == GREENLIGHT_SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = GREENLIGHT_SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = GREENLIGHT_SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == FUNNEL_SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = FUNNEL_SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = FUNNEL_SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == ASSERTIVE_SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = ASSERTIVE_SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = ASSERTIVE_SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == BALANCED_V12_SAFE_OUTSIDE_PROMPT_PROFILE:
        prompt_version = BALANCED_V12_SAFE_OUTSIDE_PROMPT_VERSION
        system_prompt = BALANCED_V12_SAFE_OUTSIDE_SYSTEM_PROMPT
    elif prompt_profile == CUSTOM_SAFE_OUTSIDE_PROMPT_PROFILE:
        if not args.system_prompt_file:
            raise ValueError(
                "--system-prompt-file is required for safe-outside-custom-file"
            )
        prompt_file = Path(args.system_prompt_file).resolve()
        if not prompt_file.is_file():
            raise FileNotFoundError(prompt_file)
        system_prompt = prompt_file.read_text(encoding="utf-8")
        prompt_source_file = str(prompt_file)
        prompt_source_sha256 = _sha256(prompt_file)
        prompt_version = (
            "candidate_island_scorer_v11_omni_preaudit_custom_"
            + prompt_source_sha256[:24]
        )
    else:
        raise ValueError(f"unknown Scorer v11 teacher prompt profile: {prompt_profile}")
    rows = _rows(manifest)
    labels_path = output / "preaudit.jsonl"
    raw_path = output / "raw_responses.jsonl"
    existing = _resume_index(
        labels_path,
        model=model,
        prompt_version=prompt_version,
    )
    pending = [row for row in rows if str(row["source_id"]) not in existing]
    if args.limit > 0:
        pending = pending[: args.limit]
    progress_total = len(existing) + len(pending)
    progress_initial_completed = len(existing)
    progress_path = output / "progress.json"
    started = time.perf_counter()
    profile_name = env_file.name
    audio_content_mode = {"qwen": "input_audio", "gemini": "input_audio_raw"}[profile_name.lower()]
    _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "model": model, "completed": len(existing), "total": progress_total, "pending": len(pending), "elapsed_s": 0.0})
    for index, row in enumerate(pending, start=1):
        source_id = str(row["source_id"])
        audio = _resolve_audio(str(row["audio"]), manifest=manifest)
        feedback = ""
        last_error: Exception | None = None
        for attempt in range(1, args.max_attempts + 1):
            parsed: dict[str, Any] | None = None
            raw: dict[str, Any] | None = None
            try:
                request_started = time.perf_counter()
                print(f"omni_request={len(existing)-progress_initial_completed+1}/{len(pending)} provider={profile_name} source_id={source_id} attempt={attempt}/{args.max_attempts}", flush=True)
                parsed, raw = call_omni(audio_path=audio, fmt=audio.suffix.lstrip(".") or "wav", audio_content_mode=audio_content_mode, model=model, api_key=api_key, base_url=base_url, timeout_s=args.timeout_s, store_stream_chunks=False, prompt=_prompt(row, feedback=feedback, prompt_profile=prompt_profile), system_prompt=system_prompt, max_tokens=args.max_tokens, enable_thinking=args.enable_thinking, thinking_budget=args.thinking_budget)
                safe_outside: list[dict[str, Any]] = []
                if prompt_profile in SAFE_OUTSIDE_PROMPT_PROFILES:
                    islands, unsure, safe_outside = _safe_outside_complement(
                        parsed,
                        duration_s=float(row["duration_s"]),
                        frame_count=int(row.get("frame_count") or 0),
                    )
                else:
                    islands, unsure = _spans(parsed, duration_s=float(row["duration_s"]))
                label = {"schema": SCHEMA, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "prompt_source_file": prompt_source_file, "prompt_source_sha256": prompt_source_sha256, "source_id": source_id, "partition": str(row.get("partition") or ""), "frame_count": int(row.get("frame_count") or 0), "frame_hop_s": FRAME_HOP_S, "audio": str(row["audio"]), "audio_sha256": str(row.get("audio_sha256") or _sha256(audio)), "model": model, "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url, "env_file_name": profile_name, "overall_confidence": _number(parsed.get("overall_confidence", 0.0), name="overall confidence"), "overall_reason": str(parsed.get("overall_reason") or ""), "islands": islands, "unsure_spans": unsure, "safe_outside_spans": safe_outside, "complement_semantics": "provisional_keep_not_confirmed_inside" if prompt_profile in SAFE_OUTSIDE_PROMPT_PROFILES else "not_applicable", "reviewed_full_source": False, "preaudit_provenance": f"omni:{model}", "human_review_required": True, "training_manifest_allowed": False, "attempts": attempt}
                with labels_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"source_id": source_id, "attempt": attempt, "parsed": parsed, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
                existing[source_id] = label
                elapsed = time.perf_counter() - request_started
                total_elapsed = time.perf_counter() - started
                run_completed = len(existing) - progress_initial_completed
                rate = run_completed / max(total_elapsed, 1e-9)
                eta = max(0.0, (progress_total - len(existing)) / max(rate, 1e-9))
                print(f"omni_candidate_island={run_completed}/{len(pending)} provider={profile_name} source_id={source_id} islands={len(islands)} safe_outside={len(safe_outside)} unsure={len(unsure)} request_s={elapsed:.1f} eta_s={eta:.0f}", flush=True)
                _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "model": model, "completed": len(existing), "total": progress_total, "pending": progress_total - len(existing), "last_source_id": source_id, "last_request_s": round(elapsed, 3), "elapsed_s": round(total_elapsed, 3), "eta_s": round(eta, 3), "islands": len(islands), "safe_outside": len(safe_outside), "unsure": len(unsure)})
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                feedback = str(error)
                print(f"omni_error provider={profile_name} source_id={source_id} attempt={attempt}/{args.max_attempts} error={type(error).__name__}: {error}", flush=True)
                _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "running", "provider_profile": profile_name, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "model": model, "completed": len(existing), "total": progress_total, "pending": progress_total - len(existing), "last_source_id": source_id, "last_error": f"{type(error).__name__}: {error}", "elapsed_s": round(time.perf_counter() - started, 3)})
                with raw_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"source_id": source_id, "attempt": attempt, "error": repr(error), "parsed": parsed, "response": raw}, ensure_ascii=False, sort_keys=True) + "\n")
                if attempt < args.max_attempts:
                    time.sleep(min(8.0, float(attempt)))
        if last_error is not None:
            duration_s = float(row["duration_s"])
            frame_count = int(row.get("frame_count") or round(duration_s / FRAME_HOP_S))
            label = {
                "schema": SCHEMA,
                "prompt_profile": prompt_profile,
                "prompt_version": prompt_version,
                "source_id": source_id,
                "partition": str(row.get("partition") or ""),
                "frame_count": frame_count,
                "frame_hop_s": FRAME_HOP_S,
                "audio": str(row["audio"]),
                "audio_sha256": str(row.get("audio_sha256") or _sha256(audio)),
                "model": model,
                "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url,
                "env_file_name": profile_name,
                "overall_confidence": 0.0,
                "overall_reason": f"teacher validation failed closed: {type(last_error).__name__}: {last_error}",
                "islands": [],
                "safe_outside_spans": [],
                "unsure_spans": [{
                    "label": "unsure",
                    "start_s": 0.0,
                    "end_s": duration_s,
                    "start_frame": 0,
                    "end_frame": frame_count,
                    "reason": "teacher request failed validation; exclude the whole source from outside truth",
                }],
                "reviewed_full_source": False,
                "preaudit_provenance": f"omni:{model}:validation_failure",
                "human_review_required": True,
                "training_manifest_allowed": False,
                "teacher_failed_closed": True,
                "attempts": args.max_attempts,
            }
            with labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")
            existing[source_id] = label
            print(
                f"omni_candidate_island={len(existing)-progress_initial_completed}/{len(pending)} provider={profile_name} "
                f"source_id={source_id} failed_closed_unsure=1",
                flush=True,
            )
        if index < len(pending) and args.request_interval_s > 0:
            time.sleep(args.request_interval_s)
    result_rows = _rows(labels_path)
    summary = {"schema": SUMMARY_SCHEMA, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "prompt_source_file": prompt_source_file, "prompt_source_sha256": prompt_source_sha256, "model": model, "env_file_name": profile_name, "audio_content_mode": audio_content_mode, "base_url_host": base_url.split("/", 3)[2] if "://" in base_url else base_url, "manifest": str(manifest), "manifest_sha256": _sha256(manifest), "source_count": len(result_rows), "labeled_count": len(result_rows), "manual_review_required": True, "training_manifest_allowed": False, "labels": str(labels_path), "raw_responses": str(raw_path)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_progress(progress_path, {"schema": "candidate_island_scorer_v11_omni_progress_v1", "status": "completed", "provider_profile": profile_name, "prompt_profile": prompt_profile, "prompt_version": prompt_version, "model": model, "completed": len(result_rows), "total": progress_total, "pending": max(0, progress_total - len(result_rows)), "elapsed_s": round(time.perf_counter() - started, 3), "summary": str(output / "summary.json")})
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        choices=(DEFAULT_PROMPT_PROFILE, *SAFE_OUTSIDE_PROMPT_PROFILES),
        help=(
            "Teacher task. dialogue-islands-v5 is the current baseline; "
            "safe-outside-complement-v1 is the verbose outside-first experiment; "
            "safe-outside-complement-v2-simple, v3-greenlight, v4-funnel, and "
            "v5-assertive and v6-balanced-v12-teacher are simplified outside-first experiments."
            " safe-outside-custom-file accepts an explicit system prompt file."
        ),
    )
    parser.add_argument(
        "--env-file",
        default="gemini",
        choices=("qwen", "gemini"),
        help="Named ~/.config/omni profile. Gemini is the default; use qwen explicitly.",
    )
    parser.add_argument("--api-key-env", default=",".join(DEFAULT_API_KEY_ENV_CANDIDATES))
    parser.add_argument("--model-env", default="OMNI_MODEL,QWEN_OMNI_MODEL")
    parser.add_argument("--base-url-env", default=",".join(DEFAULT_BASE_URL_ENV_CANDIDATES))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-interval-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--system-prompt-file",
        default="",
        help="UTF-8 system prompt file, required with --prompt-profile safe-outside-custom-file.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False))
