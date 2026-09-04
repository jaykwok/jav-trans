import { escHtml, readErrorDetail, showToast } from './util.js';
import { qcOverlay, qcTitle, qcBody, qcClose, qcCopy, qcOpenMd } from './dom.js';

// The quality report is written as JSON plus a Markdown twin, and until now the
// only way to read either was to open the file. Everything the pipeline learned
// to measure this month - cut provenance, layout break types, display linger,
// the two post-gate layers - lives in keys nobody sees. This panel groups them
// the way the pipeline produces them, marks whatever tripped a QC threshold,
// and never hides a key it does not recognise (see OTHER_GROUP_TITLE).

const OTHER_GROUP_TITLE = '其他指标';

const METRIC_GROUPS = [
  {
    title: '交付规格（TTSG）',
    note: '成品字幕本身是否合规。这一组亮红的才是必须改的。',
    metrics: [
      ['spec_cue_count', '成品字幕条数'],
      ['spec_zh_cps_over_9_count', '中文 CPS > 9 条数'],
      ['spec_zh_cps_max', '中文 CPS 峰值'],
      ['spec_duration_over_7s_count', '时长 > 7s 条数'],
      ['spec_duration_under_min_count', '时长 < 5/6s 条数'],
      ['spec_duration_under_min_share', '时长 < 5/6s 占比'],
      ['spec_gap_under_2frames_count', '间隔 < 2 帧条数'],
      ['spec_gap_under_2frames_share', '间隔 < 2 帧占比'],
      ['spec_zh_line_over_16_count', '单行 > 16 字条数'],
      ['spec_zh_lines_over_2_count', '超过 2 行条数'],
      ['spec_zh_banned_punct_count', '禁用标点（成品）'],
      ['spec_zh_raw_banned_punct_count', '禁用标点（清洗前）'],
      ['spec_ja_measured', '日文轨已计入'],
      ['spec_ja_line_over_13_count', '日文单行 > 13 字条数'],
      ['spec_ja_lines_over_2_count', '日文超过 2 行条数'],
      ['spec_ja_banned_punct_count', '日文禁用标点（成品）'],
      ['spec_ja_raw_banned_punct_count', '日文禁用标点（清洗前）'],
      ['spec_ja_cps_over_4_count', '日文 CPS > 4 条数'],
      ['spec_ja_cps_over_4_share', '日文 CPS > 4 占比'],
      ['spec_ja_cps_max', '日文 CPS 峰值'],
    ],
  },
  {
    title: '时间轴与阅读时间',
    note: '出点延伸只用已经空着的静音，起点和声学边界不动。',
    metrics: [
      ['subtitle_duration_p50_s', '时长中位数'],
      ['subtitle_duration_p90_s', '时长 P90'],
      ['subtitle_duration_p95_s', '时长 P95'],
      ['subtitle_duration_max_s', '时长最大值'],
      ['display_linger_applied_count', '出点延伸条数'],
      ['display_linger_total_s', '出点延伸总时长'],
      ['short_segment_count', '短句条数'],
      ['short_segment_ratio', '短句占比'],
      ['micro_segment_count', '极短句条数'],
      ['long_segment_count', '长句条数'],
      ['subtitle_overlap_count', '重叠条数'],
      ['subtitle_overlap_total_s', '重叠总时长'],
      ['subtitle_overlap_max_s', '单处最大重叠'],
    ],
  },
  {
    title: '切分与布局（上游来源）',
    note: '音频怎么切、cue 怎么断。断点类型分布见下方。',
    metrics: [
      ['chunk_cut_policy', '切分策略'],
      ['chunk_cut_source', '切点来源'],
      ['chunk_count', '音频块数'],
      ['chunk_cut_count', '切点数'],
      ['chunk_cut_at_pause_count', '落在停顿的切点'],
      ['chunk_cut_max_fallback_count', '兜底切点数'],
      ['chunk_cut_max_fallback_share', '兜底切点占比'],
      ['chunk_cut_pause_width_median_s', '切点停顿宽度中位数'],
      ['chunk_cut_pause_width_min_s', '切点停顿宽度最小值'],
      ['chunk_duration_median_s', '块时长中位数'],
      ['chunk_duration_min_s', '块时长最小值'],
      ['chunk_duration_max_s', '块时长最大值'],
      ['cue_plan_cue_count', '布局产出 cue 数'],
      ['layout_word_gap_cut_count', '按词间隔断句数'],
      ['layout_word_gap_cut_under_0p2s', '其中间隔 < 0.2s'],
      ['layout_word_gap_median_s', '词间隔中位数'],
      ['cue_continues_from_previous_count', '承接上一条的 cue'],
      ['cue_continues_from_previous_share', '承接上一条占比'],
      ['cue_continues_into_next_count', '延续到下一条的 cue'],
      ['vocalisation_cues_dropped', '删除的纯人声 cue'],
      ['vocalisation_runs_dropped', '删除的纯人声连段'],
      ['vocalisation_continuity_flags_cleared', '清除的连续标记'],
    ],
  },
  {
    title: '密度（观感）',
    note: '每分钟条数与滑窗密度：字幕是否一直挂在屏幕上。',
    metrics: [
      ['per_min_subtitle_count', '每分钟字幕条数'],
      ['subtitle_density_cps_threshold', 'CPS 阈值'],
      ['subtitle_density_over_4cps_count', '超阈值条数'],
      ['subtitle_density_over_4cps_ratio', '超阈值占比'],
      ['subtitle_density_max_ja_cps', '日文 CPS 峰值'],
      ['subtitle_density_p90_ja_cps', '日文 CPS P90'],
      ['subtitle_density_p95_ja_cps', '日文 CPS P95'],
      ['subtitle_density_window_10s_max_cue_count', '10s 窗口最多条数'],
      ['subtitle_density_window_10s_max_active_ratio', '10s 窗口最高占屏比'],
      ['subtitle_density_window_10s_max_cps', '10s 窗口最高 CPS'],
      ['subtitle_density_window_10s_min_gap_s', '10s 窗口最小间隔'],
      ['subtitle_density_window_10s_median_gap_s', '10s 窗口间隔中位数'],
      ['subtitle_density_window_30s_max_cue_count', '30s 窗口最多条数'],
      ['subtitle_density_window_30s_max_active_ratio', '30s 窗口最高占屏比'],
      ['subtitle_density_window_30s_max_cps', '30s 窗口最高 CPS'],
      ['subtitle_density_window_30s_min_gap_s', '30s 窗口最小间隔'],
      ['subtitle_density_window_30s_median_gap_s', '30s 窗口间隔中位数'],
    ],
  },
  {
    title: '文本与译文',
    metrics: [
      ['empty_zh_ratio', '空译文占比'],
      ['repetition_ratio', '重复译文占比'],
      ['kana_only_ratio', '纯假名占比'],
      ['glossary_hit_rate', '术语命中率'],
    ],
  },
  {
    title: 'ASR 健康度',
    metrics: [
      ['alignment_issue_count', '对齐异常数'],
      ['alignment_issue_total', '对齐检查总数'],
      ['alignment_issue_ratio', '对齐异常占比'],
      ['asr_generation_error_count', '生成错误数'],
      ['asr_generation_overflow_count', '生成溢出数'],
      ['asr_timeout_count', '超时数'],
      ['asr_quarantined_count', '隔离块数'],
    ],
  },
  {
    title: '复读检测（post-gate）',
    note: '两层都给：检测器在音频块上看到了什么，以及有多少真的活到了成品字幕里。',
    metrics: [
      ['postgate_chunks_reviewed', '受检音频块'],
      ['postgate_chunks_flagged', '被标记音频块'],
      ['postgate_chunks_flagged_share', '被标记音频块占比'],
      ['postgate_alignment_score_checked', '对齐分数受检数'],
      ['postgate_flagged_cue_count', '被标记 cue 数'],
      ['postgate_flagged_cue_share', '被标记 cue 占比'],
    ],
  },
];

// Rendered on their own, not as label/value rows.
const STRUCTURED_KEYS = new Set([
  'warnings',
  'layout_break_type_counts',
  'postgate_chunk_flag_counts',
  'postgate_cue_flag_counts',
  'spec_review_examples',
  'subtitle_density_review_examples',
  'subtitle_overlap_examples',
]);

const GROUPED_KEYS = new Set(
  METRIC_GROUPS.flatMap(group => group.metrics.map(([key]) => key)),
);

function formatValue(key, value) {
  if (value == null) return '-';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return String(value);
  if (/_(ratio|share|rate)$/.test(key)) return `${(value * 100).toFixed(1)}%`;
  if (Number.isInteger(value)) return String(value);
  const rounded = Math.round(value * 1000) / 1000;
  return /_s$/.test(key) ? `${rounded} s` : String(rounded);
}

export function timecode(seconds) {
  const total = Number(seconds);
  if (!Number.isFinite(total) || total < 0) return '-';
  const whole = Math.floor(total);
  const ms = Math.round((total - whole) * 1000);
  const hh = String(Math.floor(whole / 3600)).padStart(2, '0');
  const mm = String(Math.floor((whole % 3600) / 60)).padStart(2, '0');
  const ss = String(whole % 60).padStart(2, '0');
  return `${hh}:${mm}:${ss}.${String(ms).padStart(3, '0')}`;
}

// Every warning is "<metric>=<value> <op> <ENV>=<limit>", so the metric it came
// from is whatever sits in front of the first '='. That is what lets a warned
// row highlight itself and carry the threshold as its tooltip.
function warningsByMetric(warnings) {
  const map = new Map();
  warnings.forEach(warning => {
    const key = String(warning).split('=')[0].trim();
    if (!key) return;
    map.set(key, [...(map.get(key) || []), String(warning)]);
  });
  return map;
}

function renderMetricRows(report, metrics, warned) {
  return metrics
    .filter(([key]) => key in report)
    .map(([key, label]) => {
      const messages = warned.get(key);
      const cls = messages ? ' qc-metric-warned' : '';
      const tip = messages ? ` title="${escHtml(messages.join('\n'))}"` : ` title="${escHtml(key)}"`;
      return `<div class="qc-metric${cls}"${tip}>
                <span class="qc-metric-label">${escHtml(label)}</span>
                <span class="qc-metric-value">${escHtml(formatValue(key, report[key]))}${messages ? ' ⚠' : ''}</span>
              </div>`;
    })
    .join('');
}

function renderGroups(report, warned) {
  const sections = METRIC_GROUPS.map(group => {
    const rows = renderMetricRows(report, group.metrics, warned);
    if (!rows) return '';
    const note = group.note ? `<p class="qc-note">${escHtml(group.note)}</p>` : '';
    return `<section class="qc-section">
              <h3>${escHtml(group.title)}</h3>
              ${note}
              <div class="qc-metrics">${rows}</div>
            </section>`;
  });

  const leftovers = Object.keys(report)
    .filter(key => !GROUPED_KEYS.has(key) && !STRUCTURED_KEYS.has(key))
    .filter(key => typeof report[key] !== 'object' || report[key] === null)
    .map(key => [key, key]);
  if (leftovers.length) {
    sections.push(`<section class="qc-section">
                     <h3>${escHtml(OTHER_GROUP_TITLE)}</h3>
                     <p class="qc-note">这一版页面还没为它们写标签，先如实照搬。</p>
                     <div class="qc-metrics">${renderMetricRows(report, leftovers, warned)}</div>
                   </section>`);
  }
  return sections.filter(Boolean).join('');
}

function renderCountBars(title, counts, note = '') {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  if (!entries.length) return '';
  const total = entries.reduce((sum, [, count]) => sum + count, 0) || 1;
  const rows = entries.map(([name, count]) => {
    const pct = (count / total) * 100;
    return `<div class="qc-bar-row">
              <span class="qc-bar-name">${escHtml(name)}</span>
              <span class="qc-bar-track"><span class="qc-bar-fill" style="width:${pct.toFixed(1)}%"></span></span>
              <span class="qc-bar-value">${count} · ${pct.toFixed(1)}%</span>
            </div>`;
  }).join('');
  return `<section class="qc-section">
            <h3>${escHtml(title)}</h3>
            ${note ? `<p class="qc-note">${escHtml(note)}</p>` : ''}
            <div class="qc-bars">${rows}</div>
          </section>`;
}

function renderPostgateFlagTable(report) {
  const chunkFlags = report.postgate_chunk_flag_counts || {};
  const cueFlags = report.postgate_cue_flag_counts || {};
  const names = [...new Set([...Object.keys(chunkFlags), ...Object.keys(cueFlags)])].sort();
  if (!names.length) return '';
  const rows = names.map(name => `<tr>
      <td><code>${escHtml(name)}</code></td>
      <td class="num">${escHtml(String(chunkFlags[name] ?? 0))}</td>
      <td class="num">${escHtml(String(cueFlags[name] ?? 0))}</td>
    </tr>`).join('');
  return `<section class="qc-section">
            <h3>复读标记：音频块 vs 成品 cue</h3>
            <p class="qc-note">左列是检测器看到的，右列是删除纯人声、合并布局之后还留在字幕里的。</p>
            <table class="qc-table">
              <thead><tr><th>标记</th><th class="num">音频块</th><th class="num">成品 cue</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </section>`;
}

function renderExamples(title, note, examples, columns) {
  const rows = (examples || []).map(item => {
    const cells = columns.map(([, read]) => `<td class="num">${escHtml(read(item))}</td>`).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
  if (!rows) return '';
  const head = columns.map(([label]) => `<th class="num">${escHtml(label)}</th>`).join('');
  return `<section class="qc-section">
            <h3>${escHtml(title)}</h3>
            ${note ? `<p class="qc-note">${escHtml(note)}</p>` : ''}
            <table class="qc-table">
              <thead><tr>${head}</tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </section>`;
}

function renderReport(report) {
  const warnings = Array.isArray(report.warnings) ? report.warnings : [];
  const warned = warningsByMetric(warnings);
  const warningBlock = warnings.length
    ? `<section class="qc-section qc-warnings">
         <h3>QC 警告 (${warnings.length})</h3>
         <ul>${warnings.map(w => `<li>${escHtml(w)}</li>`).join('')}</ul>
       </section>`
    : `<section class="qc-section qc-clean"><h3>QC 警告</h3><p>没有触发任何阈值。</p></section>`;

  return [
    warningBlock,
    renderGroups(report, warned),
    renderCountBars(
      '布局断点类型',
      report.layout_break_type_counts,
      '每条 cue 是因为什么断开的：标点、强停顿，还是只能按词间隔切。',
    ),
    renderPostgateFlagTable(report),
    renderExamples(
      '规格问题样例',
      '按时间码到播放器里核对；这里只列前若干条。',
      report.spec_review_examples,
      [
        ['#', item => item.index],
        ['起点', item => timecode(item.start)],
        ['终点', item => timecode(item.end)],
        ['问题', item => (item.issues || []).join(' / ')],
      ],
    ),
    renderExamples(
      '密度问题样例',
      '日文 CPS 最高的几条，通常是过短的 cue 塞了整句话。',
      report.subtitle_density_review_examples,
      [
        ['#', item => item.index],
        ['起点', item => timecode(item.start)],
        ['时长', item => `${item.duration_s} s`],
        ['字符', item => item.ja_units],
        ['CPS', item => item.ja_cps],
      ],
    ),
    renderExamples(
      '重叠样例',
      '成品字幕里不应存在；出现即为回归。',
      report.subtitle_overlap_examples,
      [
        ['上一条', item => `${timecode(item.previous_start)} → ${timecode(item.previous_end)}`],
        ['当前条', item => `${timecode(item.current_start)} → ${timecode(item.current_end)}`],
        ['重叠', item => `${item.overlap_s} s`],
      ],
    ),
  ].filter(Boolean).join('');
}

let currentPayload = null;
let currentJobId = '';

function closeQcReport() {
  qcOverlay.hidden = true;
  currentPayload = null;
  currentJobId = '';
}

export async function openQcReport(jobId, title) {
  currentJobId = jobId;
  currentPayload = null;
  qcTitle.textContent = `质量报告 · ${title || jobId}`;
  qcOpenMd.hidden = true;
  qcBody.innerHTML = '<p class="qc-placeholder">读取中…</p>';
  qcOverlay.hidden = false;

  let payload;
  try {
    const response = await fetch(`/api/quality/${encodeURIComponent(jobId)}`);
    if (!response.ok) {
      qcBody.innerHTML = `<p class="qc-placeholder">读取失败：${escHtml(await readErrorDetail(response))}</p>`;
      return;
    }
    payload = await response.json();
  } catch (error) {
    qcBody.innerHTML = `<p class="qc-placeholder">读取失败：${escHtml(error.message)}</p>`;
    return;
  }
  if (currentJobId !== jobId) return;

  if (!payload.available) {
    currentPayload = payload;
    qcOpenMd.hidden = !payload.markdown_name;
    qcBody.innerHTML = payload.reason === 'markdown_only'
      ? `<p class="qc-placeholder">这次任务只留下了 .md 报告，没有可解析的 JSON。<br>
         用上方的「打开 .md」查看。</p>`
      : `<p class="qc-placeholder">这次任务没有质量报告。<br>
         在「参数调优 → 生成质量报告（.md）」打开开关后重新运行即可。</p>`;
    return;
  }

  currentPayload = payload;
  qcTitle.textContent = `质量报告 · ${payload.stem || title || jobId}`;
  qcOpenMd.hidden = false;
  qcBody.innerHTML = renderReport(payload.report || {});
  qcBody.scrollTop = 0;
}

export function installQcReport() {
  qcClose.addEventListener('click', closeQcReport);
  qcOverlay.addEventListener('click', e => {
    if (e.target === qcOverlay) closeQcReport();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !qcOverlay.hidden) closeQcReport();
  });
  qcCopy.addEventListener('click', async () => {
    if (!currentPayload?.report) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(currentPayload.report, null, 2));
      showToast('质量报告 JSON 已复制');
    } catch (error) {
      showToast('复制失败：' + error.message);
    }
  });
  qcOpenMd.addEventListener('click', async () => {
    if (!currentPayload?.markdown_name || !currentJobId) return;
    try {
      const response = await fetch(
        `/api/open-artifact?job_id=${encodeURIComponent(currentJobId)}&path=${encodeURIComponent(currentPayload.markdown_name)}`,
        { method: 'POST' },
      );
      if (!response.ok) showToast('打开失败：' + await readErrorDetail(response));
    } catch (error) {
      showToast('打开失败：' + error.message);
    }
  });
}
