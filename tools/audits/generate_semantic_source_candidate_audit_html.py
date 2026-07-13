#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _expectations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in dict(payload).items()}


def build_audit(
    *,
    labels: Path,
    output_dir: Path,
    semantic_expectations: Path | None = None,
) -> Path:
    rows = _rows(labels)
    expectations = _expectations(semantic_expectations)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload_rows = []
    for row in rows:
        sample_dir = audio_dir / str(row["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        source = Path(row["audio"])
        full_audio = sample_dir / f"full{source.suffix}"
        shutil.copy2(source, full_audio)
        candidates = []
        for candidate in row["candidates"]:
            copied = dict(candidate)
            for field in ("original_audio", "marked_audio"):
                candidate_source = Path(candidate[field])
                destination = sample_dir / candidate_source.name
                shutil.copy2(candidate_source, destination)
                copied[field] = destination.relative_to(output_dir).as_posix()
            candidates.append(copied)
        payload_rows.append(
            {
                **row,
                "audio": full_audio.relative_to(output_dir).as_posix(),
                "candidates": candidates,
                "sample_semantic_expectation": expectations.get(
                    str(row["sample_id"]), ""
                ),
            }
        )
    payload = json.dumps(payload_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Source Candidate Smoke</title>
<style>body{{margin:0;background:#f3f5f7;color:#20242a;font-family:Segoe UI,Arial,sans-serif}}header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid #ccd2d9;padding:12px 18px}}main{{max-width:1180px;margin:18px auto;padding:0 14px}}.help,article{{background:#fff;border:1px solid #ccd2d9;border-radius:7px;padding:16px;margin-bottom:16px}}.help{{border-left:5px solid #1769aa}}.legend{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:8px}}.legend div,.candidate{{border:1px solid #d8dde3;border-radius:6px;padding:10px}}.semantic_target{{background:#d9f2df}}.discardable{{background:#eceff2}}.unsure{{background:#fff0c8}}.layers{{display:grid;gap:8px;margin:12px 0}}.layer{{padding:10px;border:1px solid #c9d2dc;border-radius:6px;background:#f7f9fb}}.sample-note{{border-left:5px solid #8a5a00;background:#fff6dc}}details{{margin-top:12px;border:1px solid #ccd2d9;border-radius:6px;padding:10px}}summary{{cursor:pointer;font-weight:600}}.candidates{{display:grid;gap:10px;margin-top:12px}}.candidate{{display:grid;grid-template-columns:90px minmax(210px,1fr) minmax(210px,1fr) 170px minmax(220px,1fr);gap:10px;align-items:center}}audio{{width:100%;min-width:180px}}input,select,button{{font:inherit;padding:6px}}input.reason{{width:95%}}.verdict{{margin-top:12px;padding:10px;background:#f6f8fa}}button.active{{background:#1769aa;color:#fff}}article.done{{border-left:5px solid #197642}}small{{color:#59616c}}.marker{{font-weight:600}}@media(max-width:900px){{.candidate{{grid-template-columns:1fr}}.legend{{grid-template-columns:1fr}}}}</style></head><body><header><strong>1.7B Semantic Speech · 证据点与 chunk 职责分层审计</strong>　<button id="save">保存全部裁决</button> <span id="status"></span></header><main>
<section class="help"><h2>每一行在判断什么</h2><ol><li>先用卡片顶部播放器听完整隔离 source utterance，并核对固定参考文本。参考文本只帮助区分词句和非词 vocalization，不提供位置。</li><li>每行“原始候选邻域”是不加改动的当前声学邻域；“1 秒静音标记版”在精确候选帧插入了 1 秒静音。两者可独立播放。</li><li><strong>只判断静音标记紧邻位置，不判断整段邻域。</strong>即使参考文本含有词语，标记处若只有喘息、呻吟、亲吻声、笑声或非词叫声，也必须选 discardable；标记切入/紧贴清楚词句才选 semantic_target；转换边缘或听不清选 unsure。</li><li><strong>候选行不是 chunk。</strong>content 标签只作为语义证据；完整 source 的 membership head 决定 coarse island。contains_semantic source 先保持一个 island，再由 Outer 只收首尾 paired edges；内部 discardable 不会把 chunk 切碎。</li><li>这些是学习型表示选出的当前候选，不显示旧切点，也不要求标签单调。页面不允许修改时间，因为本轮 teacher 不产生 timing。</li><li>逐行修正后再判断“分类 + 分组”是否通过；5/5 通过前不会扩大或训练。</li></ol><div class="legend"><div class="semantic_target"><b>semantic_target</b><br>标记切入/紧贴清楚可辨、有字幕价值的词句</div><div class="discardable"><b>discardable</b><br>标记处仅噪声/喘息/呻吟/亲吻声/非词 vocalization</div><div class="unsure"><b>unsure</b><br>转换边缘、疑似词语、重叠或无法可靠判断</div></div></section><div id="list"></div></main>
<script>const rows={payload};const key='semantic-source-candidate-smoke5-v4:'+location.pathname;const ann=JSON.parse(localStorage.getItem(key)||'{{}}');let active=null;
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function ensure(row){{ann[row.sample_id]??={{candidates:row.candidates.map(x=>({{...x}})),verdict:'',note:''}};return ann[row.sample_id];}}
function persist(){{localStorage.setItem(key,JSON.stringify(ann));status();}}
function status(){{const done=rows.filter(r=>ensure(r).verdict==='approve').length;document.getElementById('status').textContent=`人工通过 ${{done}} / ${{rows.length}}`;}}
function contentRuns(candidates){{let runs=0,active=false;for(const c of candidates){{const retained=c.label!=='discardable';if(retained&&!active)runs++;active=retained;}}return runs;}}
function grouping(row,candidates){{const runs=contentRuns(candidates);if(row.source_gate.label==='discardable')return `content run=${{runs}}；membership 输出 0 个 island`;if(row.source_gate.label==='unsure')return `content run=${{runs}}；membership abstain`;return `content run=${{runs}}；membership 输出 1 个 coarse island（不会按 run 切碎）`;}}
function bindAudio(root){{root.querySelectorAll('audio').forEach(audio=>audio.onplay=()=>{{if(active&&active!==audio)active.pause();active=audio;}});}}
function render(){{const root=document.getElementById('list');root.innerHTML='';for(const row of rows){{const a=ensure(row);const card=document.createElement('article');if(a.verdict==='approve')card.classList.add('done');const expectation=row.sample_semantic_expectation?`<div class="layer sample-note"><b>Layer 3 · 本样本 Split 期望（只约束本例）</b><br>${{esc(row.sample_semantic_expectation)}}</div>`:`<div class="layer"><b>Layer 3 · 最终 Split</b><br>本页不裁决该样本的最终语义 chunk 粒度。</div>`;card.innerHTML=`<h3>${{esc(row.sample_id)}}</h3><p><b>审计重点：</b>${{esc(row.audit_focus)}}</p><p><b>固定参考文本：</b>${{esc(row.reference_text)}}</p><small>来源：${{esc(row.source)}}；完整时长 ${{Number(row.duration_s).toFixed(3)}}s；候选选择：learned hidden farthest-medoid</small><div class="layers"><div class="layer"><b>Layer 1 · content 证据</b><br>逐点分类只描述当前位置，不产生 chunk。</div><div class="layer"><b>Layer 2 · source membership / Outer</b><br>${{esc(grouping(row,a.candidates))}}；source gate=${{esc(row.source_gate.label)}}（${{Number(row.source_gate.confidence).toFixed(2)}}）</div>${{expectation}}</div><p><b>完整原音频</b></p><audio controls preload="none" src="${{esc(row.audio)}}"></audio><details><summary>展开 ${{a.candidates.length}} 个 content 证据点（不是 chunk）</summary><div class="candidates">${{a.candidates.map((c,i)=>`<div class="candidate ${{esc(c.label)}}" data-index="${{i}}"><div class="marker">证据点 ${{esc(c.candidate_id)}}<br><small>${{esc(c.label_source)}}</small></div><div><small>原始候选邻域（非 chunk）</small><audio controls preload="none" src="${{esc(c.original_audio)}}"></audio></div><div><small>1 秒静音标记版</small><audio controls preload="none" src="${{esc(c.marked_audio)}}"></audio></div><select data-field="label">${{['semantic_target','discardable','unsure'].map(v=>`<option ${{c.label===v?'selected':''}}>${{v}}</option>`).join('')}}</select><input class="reason" data-field="reason" value="${{esc(c.reason)}}"></div>`).join('')}}</div></details><div class="verdict"><b>本样本：</b> <button data-verdict="approve" class="${{a.verdict==='approve'?'active':''}}">分类与分组通过</button> <button data-verdict="reject" class="${{a.verdict==='reject'?'active':''}}">仍不通过</button><br><input class="reason note" placeholder="备注" value="${{esc(a.note)}}"></div>`;card.querySelectorAll('.candidate').forEach(el=>{{const i=Number(el.dataset.index);el.querySelectorAll('[data-field]').forEach(input=>input.onchange=()=>{{a.candidates[i][input.dataset.field]=input.value;persist();render();}});}});card.querySelectorAll('[data-verdict]').forEach(b=>b.onclick=()=>{{a.verdict=b.dataset.verdict;a.updated_at=new Date().toISOString();persist();render();}});card.querySelector('.note').onchange=e=>{{a.note=e.target.value;a.updated_at=new Date().toISOString();persist();}};bindAudio(card);root.appendChild(card);}}status();}}
document.getElementById('save').onclick=async()=>{{const content=rows.map(row=>{{const a=ensure(row);return JSON.stringify({{schema:'semantic_source_candidate_manual_verdict_v4',sample_id:row.sample_id,audio:row.audio,duration_s:row.duration_s,reference_text:row.reference_text,source_gate:row.source_gate,grouping_preview:grouping(row,a.candidates),sample_semantic_expectation:row.sample_semantic_expectation,candidates:a.candidates,verdict:a.verdict||'unreviewed',note:a.note||'',updated_at:a.updated_at||new Date().toISOString()}});}}).join('\\n')+'\\n';const res=await fetch('/__audit_api__/save-labels',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{href:location.pathname,filename:'manual_verdicts.jsonl',content}})}});const out=await res.json();document.getElementById('status').textContent=out.ok?'已保存到 '+out.path:'保存失败: '+out.error;}};render();</script></body></html>"""
    index = output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate semantic source learned-candidate audit."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--semantic-expectations", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        build_audit(
            labels=Path(args.labels),
            output_dir=Path(args.output_dir),
            semantic_expectations=(
                Path(args.semantic_expectations) if args.semantic_expectations else None
            ),
        )
    )
