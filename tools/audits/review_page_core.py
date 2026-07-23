"""Reusable shell, playback, state, and save API for human audit pages."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import html
from itertools import product


AUDIO_SPAN_PLAYER_JS = r"""
let activeAudio=null,activeButton=null,stopFn=null,playToken=0;
function stop(){
  playToken+=1;
  if(activeAudio&&stopFn)activeAudio.removeEventListener('timeupdate',stopFn);
  if(activeAudio)activeAudio.pause();
  if(activeButton)activeButton.classList.remove('playing');
  activeAudio=activeButton=stopFn=null;
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
  const token=++playToken;
  activeAudio=audio;
  activeButton=button;
  button.classList.add('playing');
  const status=document.getElementById('status');
  status.textContent=`加载 ${start.toFixed(2)}–${end.toFixed(2)}s…`;
  try{
    await waitForMetadata(audio);
    if(token!==playToken)return;
    if(!Number.isFinite(audio.duration)||audio.duration<=0)throw new Error('invalid audio duration');
    const safeStart=Math.max(0,Math.min(start,Math.max(0,audio.duration-.001)));
    const safeEnd=Math.max(safeStart,Math.min(end,audio.duration));
    audio.currentTime=safeStart;
    stopFn=()=>{
      if(token===playToken&&audio.currentTime>=safeEnd-.005){
        status.textContent=`已停止 ${safeStart.toFixed(2)}–${safeEnd.toFixed(2)}s`;
        stop();
      }
    };
    audio.addEventListener('timeupdate',stopFn);
    await audio.play();
    if(token!==playToken){audio.pause();return;}
    status.textContent=`播放 ${safeStart.toFixed(2)}–${safeEnd.toFixed(2)}s`;
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
    button.title=config.title?config.title(span,start,end):`${config.label} ${start.toFixed(2)}–${end.toFixed(2)}s`;
    button.textContent=config.text?config.text(span,start,end):`${start.toFixed(2)}–${end.toFixed(2)}s`;
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
    button.textContent=`${Number(span.start_s).toFixed(2)}–${Number(span.end_s).toFixed(2)}s`;
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
    const content=config.entries.map(entry=>JSON.stringify(config.serialize(entry,ensure(entry)))).join('\n')+'\n';
    try{
      const response=await fetch('/__audit_api__/save-labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({href:location.pathname,filename:config.filename||'manual_verdicts.jsonl',content})});
      const result=await response.json();
      updateStatus(response.ok&&result.ok?'已保存到 '+result.path:'保存失败: '+(result.error||response.status));
    }catch(error){updateStatus('保存失败: '+error.message);}
  }
  return {ensure,persist,save,updateStatus,annotations};
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
