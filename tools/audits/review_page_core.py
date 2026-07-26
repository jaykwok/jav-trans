"""Reusable shell, playback, state, and save API for human audit pages."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import html
from itertools import product

from tools.omni.timestamp_contract import format_mmss_timestamp


def format_audit_timestamp(seconds: float) -> str:
    """Format audit labels with the same precision as the teacher wire contract."""

    return format_mmss_timestamp(seconds)


AUDIO_SPAN_PLAYER_JS = r"""
let activeAudio=null,activeButton=null,stopFn=null,stopTimer=null,stopFrame=null,playToken=0;
function formatAuditTimestamp(value){
  const numeric=Number(value);
  if(!Number.isFinite(numeric)||numeric<0)return '--:--.---';
  const totalMilliseconds=Math.round(numeric*1000);
  const milliseconds=totalMilliseconds%1000;
  const totalSeconds=Math.floor(totalMilliseconds/1000);
  const seconds=totalSeconds%60;
  const minutes=Math.floor(totalSeconds/60);
  return `${String(minutes).padStart(2,'0')}:${String(seconds).padStart(2,'0')}.${String(milliseconds).padStart(3,'0')}`;
}
function formatAuditSpan(start,end){return `${formatAuditTimestamp(start)}–${formatAuditTimestamp(end)}`;}
function stop(finalTime=null){
  playToken+=1;
  if(activeAudio&&stopFn){
    activeAudio.removeEventListener('timeupdate',stopFn);
    activeAudio.removeEventListener('ended',stopFn);
  }
  if(stopTimer!==null)clearTimeout(stopTimer);
  if(stopFrame!==null)cancelAnimationFrame(stopFrame);
  if(activeAudio){
    activeAudio.pause();
    if(Number.isFinite(finalTime))activeAudio.currentTime=finalTime;
  }
  if(activeButton)activeButton.classList.remove('playing');
  activeAudio=activeButton=stopFn=null;
  stopTimer=stopFrame=null;
}
function waitForMetadata(audio){
  if(audio.readyState>=1)return Promise.resolve();
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{
      audio.removeEventListener('loadedmetadata',loaded);
      audio.removeEventListener('error',failed);
    };
    const loaded=()=>{cleanup();resolve();};
    const failed=()=>{cleanup();reject(audio.error||new Error('audio metadata load failed'));};
    audio.addEventListener('loadedmetadata',loaded,{once:true});
    audio.addEventListener('error',failed,{once:true});
    audio.load();
  });
}
async function play(audio,button,start,end){
  stop();
  if(!Number.isFinite(start)||!Number.isFinite(end)||end<=start){
    document.getElementById('status').textContent='播放区间无效';
    return;
  }
  const token=++playToken;
  activeAudio=audio;
  activeButton=button;
  button.classList.add('playing');
  const status=document.getElementById('status');
  status.textContent=`加载 ${formatAuditSpan(start,end)}…`;
  try{
    await waitForMetadata(audio);
    if(token!==playToken)return;
    if(!Number.isFinite(audio.duration)||audio.duration<=0)throw new Error('invalid audio duration');
    const safeStart=Math.max(0,Math.min(start,Math.max(0,audio.duration-.001)));
    const safeEnd=Math.max(safeStart,Math.min(end,audio.duration));
    audio.currentTime=safeStart;
    const finishAtEnd=()=>{
      if(token!==playToken||activeAudio!==audio||activeButton!==button)return;
      status.textContent=`已停止 ${formatAuditSpan(safeStart,safeEnd)}`;
      stop(safeEnd);
    };
    stopFn=()=>{
      if(audio.ended||audio.currentTime>=safeEnd-.003)finishAtEnd();
    };
    audio.addEventListener('timeupdate',stopFn);
    audio.addEventListener('ended',stopFn);
    const watch=()=>{
      if(token!==playToken||activeAudio!==audio)return;
      if(audio.currentTime>=safeEnd-.003){finishAtEnd();return;}
      stopFrame=requestAnimationFrame(watch);
    };
    const started=audio.play();
    stopFrame=requestAnimationFrame(watch);
    await started;
    if(token!==playToken){audio.pause();return;}
    const playbackRate=Math.max(.01,Math.abs(Number(audio.playbackRate)||1));
    const remainingMilliseconds=Math.max(0,(safeEnd-audio.currentTime)*1000/playbackRate);
    stopTimer=setTimeout(finishAtEnd,remainingMilliseconds);
    status.textContent=`播放 ${formatAuditSpan(safeStart,safeEnd)}`;
  }catch(error){
    if(token!==playToken)return;
    status.textContent=`播放失败：${error&&error.message?error.message:String(error)}`;
    stop();
  }
}
"""


AUDIT_REVIEW_CORE_JS = AUDIO_SPAN_PLAYER_JS + r"""
function escapeAuditHtml(value){return String(value??'').replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));}
function appendAuditSpanLane(config){
  const line=document.createElement('div');
  line.className='audit-lane';
  line.innerHTML=`<div class="audit-lane-label"><b>${escapeAuditHtml(config.label)}</b><small>${escapeAuditHtml(config.metric||'')}</small></div><div class="audit-track"></div>`;
  const track=line.querySelector('.audit-track');
  for(const span of config.spans){
    const button=document.createElement('button'),start=Number(span.start_s),end=Number(span.end_s);
    button.className=`audit-span ${typeof config.className==='function'?config.className(span):config.className||''}`;
    button.style.left=`${100*start/config.durationS}%`;
    button.style.width=`${Math.max(.12,100*(end-start)/config.durationS)}%`;
    button.title=config.title?config.title(span,start,end):`${config.label} ${formatAuditSpan(start,end)}`;
    button.textContent=config.text?config.text(span,start,end):formatAuditSpan(start,end);
    button.onclick=()=>play(config.audio,button,start,end);
    track.appendChild(button);
  }
  config.container.appendChild(line);
  return line;
}
function appendAuditClipButtons(config){
  const row=document.createElement('div');
  row.className='audit-clip-row';
  row.innerHTML=`<small>${escapeAuditHtml(config.label)}：</small>`;
  if(!config.spans.length)row.innerHTML+='<small>无</small>';
  for(const span of config.spans){
    const button=document.createElement('button');
    button.type='button';
    button.className=`audit-clip-button ${config.className||''}`;
    button.textContent=formatAuditSpan(span.start_s,span.end_s);
    button.onclick=()=>play(config.audio,button,Number(span.start_s),Number(span.end_s));
    row.appendChild(button);
  }
  config.container.appendChild(row);
  return row;
}
function createAuditReviewCore(config){
  let annotations={};
  try{annotations=JSON.parse(localStorage.getItem(config.storageKey)||'{}');}
  catch(_error){annotations={};}
  function ensure(entry){
    const id=config.entryId(entry);
    if(!annotations[id])annotations[id]=JSON.parse(JSON.stringify(config.defaultState(entry)));
    return annotations[id];
  }
  function completedCount(){return config.entries.filter(entry=>config.isComplete(ensure(entry),entry)).length;}
  function updateStatus(message=''){
    const extra=config.statusExtra?config.statusExtra():'';
    document.getElementById('status').textContent=(message?message+' · ':'')+`${config.statusLabel} ${completedCount()}/${config.entries.length}${extra?' · '+extra:''}`;
  }
  function persist(){localStorage.setItem(config.storageKey,JSON.stringify(annotations));updateStatus();}
  async function save(){
    const entries=config.entries.filter(entry=>!config.shouldSerialize||config.shouldSerialize(ensure(entry),entry));
    try{
      const serialized=await Promise.all(entries.map(entry=>Promise.resolve(config.serialize(entry,ensure(entry)))));
      const lines=serialized.map(value=>JSON.stringify(value));
      const content=lines.length?lines.join('\n')+'\n':'';
      const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:config.filename||'manual_verdicts.jsonl',content})});
      const result=await response.json();
      updateStatus(response.ok&&result.ok?'已保存到 '+result.path:'保存失败: '+(result.error||response.status));
    }catch(error){updateStatus('保存失败: '+error.message);}
  }
  return {ensure,persist,save,updateStatus,annotations,isComplete:config.isComplete,shouldSerialize:config.shouldSerialize};
}

/*
 * Shared editable complete-partition helpers.  Adapters may keep their own
 * state shape, but should use these helpers rather than silently inventing
 * different boundary semantics.  A partition is frame based and uses the
 * three central labels; gaps are allowed while editing and rejected at save.
 */
const AUDIT_PARTITION_LABELS=new Set(['vocal_candidate','non_vocal_candidate','unsure']);
function cloneAuditPartition(segments){
  return (Array.isArray(segments)?segments:[]).map((segment,index)=>({
    id:String(segment.id||`segment-${index}`),
    label:String(segment.label||'unsure'),
    start_frame:Number(segment.start_frame),
    end_frame:Number(segment.end_frame),
    category:segment.category==null?'':String(segment.category),
    reason:segment.reason==null?'':String(segment.reason)
  }));
}
function normalizeAuditPartition(segments){
  const sorted=cloneAuditPartition(segments).filter(segment=>Number.isFinite(segment.start_frame)&&Number.isFinite(segment.end_frame)&&segment.end_frame>segment.start_frame).sort((a,b)=>a.start_frame-b.start_frame||a.end_frame-b.end_frame||a.id.localeCompare(b.id));
  const result=[];
  for(const segment of sorted){
    const previous=result[result.length-1];
    if(previous&&previous.label===segment.label&&previous.end_frame===segment.start_frame){
      previous.end_frame=segment.end_frame;
      previous.reason=previous.reason||segment.reason;
      previous.category=previous.category||segment.category;
    }else result.push(segment);
  }
  return result;
}
function validateAuditPartition(segments,frameCount){
  const count=Number(frameCount),normalized=normalizeAuditPartition(segments);
  if(!Number.isInteger(count)||count<=0)return {ok:false,error:'无效的 frame_count'};
  if(!normalized.length)return {ok:false,error:'至少需要一个区间'};
  let cursor=0;
  for(const segment of normalized){
    if(!AUDIT_PARTITION_LABELS.has(segment.label))return {ok:false,error:`未知标签：${segment.label}`};
    if(!Number.isInteger(segment.start_frame)||!Number.isInteger(segment.end_frame))return {ok:false,error:'区间边界必须对齐到 frame'};
    if(segment.start_frame<0||segment.end_frame>count||segment.end_frame<=segment.start_frame)return {ok:false,error:'存在越界或空区间'};
    if(segment.start_frame<cursor)return {ok:false,error:'区间重叠'};
    if(segment.start_frame>cursor)return {ok:false,error:`存在空洞：${cursor}–${segment.start_frame} frame`};
    cursor=segment.end_frame;
  }
  if(cursor!==count)return {ok:false,error:`末尾未覆盖到 ${count} frame`};
  return {ok:true,segments:normalized};
}
function auditPartitionCanonicalRows(segments){
  return normalizeAuditPartition(segments).map(segment=>({label:segment.label,start_frame:Number(segment.start_frame),end_frame:Number(segment.end_frame)}));
}
function auditPartitionCanonicalJson(segments){
  return JSON.stringify(auditPartitionCanonicalRows(segments));
}
async function auditPartitionSha256(segments){
  const bytes=new TextEncoder().encode(auditPartitionCanonicalJson(segments));
  if(!globalThis.crypto||!globalThis.crypto.subtle)throw new Error('当前浏览器不支持 Web Crypto，无法生成 corrected_span_signature');
  const digest=await globalThis.crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('');
}
function auditPartitionSeconds(segment,frameHop){
  return {start_s:Number((Number(segment.start_frame)*Number(frameHop)).toFixed(6)),end_s:Number((Number(segment.end_frame)*Number(frameHop)).toFixed(6))};
}
"""


CORE_CSS = """
body{margin:0;background:#f4f7fa;color:#18212b;font-family:Segoe UI,Microsoft YaHei,sans-serif}
header{position:sticky;top:0;z-index:3;display:flex;gap:10px;align-items:center;background:#122233;color:#fff;padding:12px 18px}
header #status{margin-left:auto}header button{padding:7px 12px}
main{max-width:1500px;margin:auto;padding:16px}
audio{width:100%;margin:8px 0}
button.playing{outline:3px solid #111;outline-offset:-3px}
small{color:#607080}
.audit-lane{display:grid;grid-template-columns:220px 1fr;gap:8px;align-items:center;margin:8px 0}
.audit-lane-label{display:flex;flex-direction:column;gap:2px}
.audit-track{position:relative;height:40px;background:#e7ebef;border-radius:5px;overflow:hidden}
.audit-span{position:absolute;top:0;height:100%;border:0;min-width:2px;cursor:pointer;font-size:10px;overflow:hidden;white-space:nowrap}
.audit-clip-row{margin-top:8px}
.audit-clip-button{border:0;border-radius:4px;padding:5px 7px;margin:3px 3px 0 0;cursor:pointer}
@media(max-width:900px){.audit-lane{grid-template-columns:1fr}}
"""


@dataclass(frozen=True)
class AuditReviewPageSpec:
    title: str
    intro_html: str
    body_html: str
    adapter_css: str
    adapter_js: str


@dataclass(frozen=True)
class AuditOptionAxis:
    field: str
    options: tuple[str, ...]


def validate_audit_option_contract(
    *,
    axes: tuple[AuditOptionAxis, ...],
    combination_results: Mapping[tuple[str, ...], str],
    is_valid_combination: Callable[[tuple[str, ...]], bool] | None = None,
) -> None:
    """Require adapters to enumerate every logically valid outcome."""

    if not axes or len({axis.field for axis in axes}) != len(axes):
        raise ValueError("audit option axes must be non-empty and uniquely named")
    for axis in axes:
        if not axis.options or len(set(axis.options)) != len(axis.options):
            raise ValueError(f"audit axis has empty or duplicate options: {axis.field}")
    if not combination_results:
        raise ValueError("audit option contract requires complete combinations")
    validity = is_valid_combination or (lambda _combination: True)
    expected = {
        combination
        for combination in product(*(axis.options for axis in axes))
        if validity(combination)
    }
    if not expected:
        raise ValueError("audit option contract has no valid combinations")
    actual = set(combination_results)
    missing_combinations = expected - actual
    unexpected_combinations = actual - expected
    if missing_combinations:
        raise ValueError(
            "audit option contract is missing valid combinations: "
            f"{sorted(missing_combinations)}"
        )
    if unexpected_combinations:
        raise ValueError(
            "audit option contract contains invalid combinations: "
            f"{sorted(unexpected_combinations)}"
        )
    used = [set() for _axis in axes]
    for combination, result in combination_results.items():
        if len(combination) != len(axes):
            raise ValueError("audit combination does not match axis count")
        if not result or result == "unreviewed":
            raise ValueError("completed audit combinations need a decisive result")
        for index, (value, axis) in enumerate(zip(combination, axes)):
            if value not in axis.options:
                raise ValueError(
                    f"unknown audit option for {axis.field}: {value}"
                )
            used[index].add(value)
    for axis, reachable in zip(axes, used):
        missing = set(axis.options) - reachable
        if missing:
            raise ValueError(
                f"audit axis has unreachable options: {axis.field}: {sorted(missing)}"
            )
    if not any("unsure" in value for axis in axes for value in axis.options):
        raise ValueError("audit option contract must expose an unsure path")


def render_audit_review_page(spec: AuditReviewPageSpec) -> str:
    """Render a shared audit shell around one task-specific adapter."""

    title = html.escape(spec.title)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{CORE_CSS}{spec.adapter_css}</style></head><body><header><b>{title}</b><button id="stop" type="button">停止播放</button><button id="save" type="button">保存裁决</button><span id="status"></span></header><main>{spec.intro_html}{spec.body_html}</main><script>{AUDIT_REVIEW_CORE_JS}{spec.adapter_js}</script></body></html>"""
