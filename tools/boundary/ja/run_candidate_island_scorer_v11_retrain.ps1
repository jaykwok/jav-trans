param(
    [ValidateSet('prepare', 'extract', 'features', 'smoke', 'full', 'gate', 'all')]
    [string]$Stage = 'prepare',
    [string]$RunRoot = '',
    [string]$AuditDir = 'agents/audits/20260722_213831_scorer-v11-train-real-source-gemini-v5-editable25',
    [string]$HeldoutManualVerdicts = 'agents/audits/20260722_095311_scorer-v11-heldout-human-omni-semantic-units-editable24/manual_verdicts.jsonl',
    [string]$BaseCanonicalSummary = 'agents/temp/20260722_183600_scorer-v11-no-tile-real-outside-canonical-final/summary.json',
    [string]$PriorRawFeatureManifest = 'agents/temp/20260722_184400_scorer-v11-no-tile-raw-features-full/raw_feature_manifest.jsonl',
    [string]$ModelPath = 'models/jaykwok-Qwen3-ASR-1.7B-JA-Anime-Galgame-hf',
    [int]$MaxPaddedFrames = 1000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $ProjectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:UV_CACHE_DIR = (Resolve-Path '.uv-cache').Path

if (-not (Test-Path -LiteralPath '.venv')) {
    throw 'Project .venv is required. Run uv venv in the project root first.'
}
if ($MaxPaddedFrames -le 0) {
    throw '-MaxPaddedFrames must be positive.'
}
if (-not $RunRoot) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $RunRoot = "agents/temp/${stamp}_scorer-v11-real-full-source-retrain"
}
$RunRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $RunRoot))
$StatePath = Join-Path $RunRoot 'pipeline_state.json'

function Invoke-UvPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    Write-Host ('uv run python ' + ($Arguments -join ' '))
    & uv run python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed with exit code $LASTEXITCODE"
    }
}

function Read-Json {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required JSON file is missing: $Path"
    }
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}

function Write-State {
    param([Parameter(Mandatory = $true)][hashtable]$State)
    New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding utf8
}

function Require-State {
    return Read-Json -Path $StatePath
}

function Invoke-Prepare {
    $manualVerdicts = Join-Path $AuditDir 'manual_verdicts.jsonl'
    if (-not (Test-Path -LiteralPath $manualVerdicts)) {
        throw "Human train verdicts are missing: $manualVerdicts"
    }
    $auditSummary = Join-Path $AuditDir 'summary.json'
    $auditManifest = Join-Path $AuditDir 'audit_manifest.jsonl'
    $base = Read-Json -Path $BaseCanonicalSummary
    foreach ($field in @(
        'synthetic_train_sources',
        'real_train_outside_sources',
        'source_windows',
        'partition_manifest'
    )) {
        if (-not $base.$field) {
            throw "Base canonical summary is missing $field"
        }
    }

    $manualDir = Join-Path $RunRoot '01-real-train-manual'
    $canonicalDir = Join-Path $RunRoot '02-canonical'
    $rebindDir = Join-Path $RunRoot '03-raw-rebind'
    Invoke-UvPython @(
        'tools/boundary/ja/compile_candidate_island_scorer_v11_real_train_manual.py',
        '--audit-summary', $auditSummary,
        '--audit-manifest', $auditManifest,
        '--manual-verdicts', $manualVerdicts,
        '--output-dir', $manualDir
    )
    $manualSources = Join-Path $manualDir 'real_train_manual_sources.jsonl'
    Invoke-UvPython @(
        'tools/boundary/ja/compile_candidate_island_scorer_v11_canonical.py',
        '--synthetic-train-sources', [string]$base.synthetic_train_sources,
        '--real-train-outside-sources', [string]$base.real_train_outside_sources,
        '--real-train-manual-sources', $manualSources,
        '--source-windows', [string]$base.source_windows,
        '--partition-manifest', [string]$base.partition_manifest,
        '--manual-verdicts', $HeldoutManualVerdicts,
        '--output-dir', $canonicalDir
    )
    $canonicalSources = Join-Path $canonicalDir 'canonical_sources.jsonl'
    Invoke-UvPython @(
        'tools/boundary/ja/rebind_candidate_island_scorer_v11_raw_features.py',
        '--canonical-sources', $canonicalSources,
        '--prior-raw-feature-manifest', $PriorRawFeatureManifest,
        '--output-dir', $rebindDir
    )
    Write-State @{
        schema = 'candidate_island_scorer_v11_retrain_pipeline_state_v1'
        run_root = $RunRoot
        stage = 'prepared'
        model_path = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $ModelPath))
        max_padded_frames = $MaxPaddedFrames
        canonical_sources = $canonicalSources
        raw_rebind_dir = $rebindDir
        missing_canonical_sources = (Join-Path $rebindDir 'missing_canonical_sources.jsonl')
        raw_feature_manifest = (Join-Path $rebindDir 'raw_feature_manifest.jsonl')
        features_dir = (Join-Path $RunRoot '05-features')
        smoke_dir = (Join-Path $RunRoot '06-smoke')
        full_dir = (Join-Path $RunRoot '07-full')
        gate_dir = (Join-Path $RunRoot '08-gate')
    }
}

function Invoke-Extract {
    $state = Require-State
    $rebindSummary = Read-Json -Path (Join-Path $state.raw_rebind_dir 'summary.json')
    if ([int]$rebindSummary.missing_source_count -eq 0) {
        Write-Host 'No new raw PTM2048 sources require extraction.'
        return
    }
    $newRawDir = Join-Path $RunRoot '04-raw-new'
    Invoke-UvPython @(
        'tools/boundary/ja/extract_candidate_island_scorer_v11_raw_features.py',
        '--canonical-sources', [string]$state.missing_canonical_sources,
        '--model-path', [string]$state.model_path,
        '--output-dir', $newRawDir,
        '--device', 'cuda',
        '--dtype', 'bfloat16',
        '--attention', 'sdpa',
        '--memory-log-every', '5',
        '--summary-every', '5'
    )
    Invoke-UvPython @(
        'tools/boundary/ja/rebind_candidate_island_scorer_v11_raw_features.py',
        '--canonical-sources', [string]$state.canonical_sources,
        '--prior-raw-feature-manifest', $PriorRawFeatureManifest,
        '--new-raw-feature-manifest', (Join-Path $newRawDir 'raw_feature_manifest.jsonl'),
        '--output-dir', [string]$state.raw_rebind_dir
    )
    $updated = @{}
    foreach ($property in $state.PSObject.Properties) {
        $updated[$property.Name] = $property.Value
    }
    $updated.stage = 'raw_features_complete'
    Write-State $updated
}

function Invoke-Features {
    $state = Require-State
    if (-not (Test-Path -LiteralPath $state.raw_feature_manifest)) {
        throw "Final raw feature manifest is missing: $($state.raw_feature_manifest)"
    }
    Invoke-UvPython @(
        'tools/boundary/ja/compile_candidate_island_scorer_v11_features.py',
        '--canonical-sources', [string]$state.canonical_sources,
        '--raw-feature-manifest', [string]$state.raw_feature_manifest,
        '--output-dir', [string]$state.features_dir
    )
}

function Invoke-Smoke {
    $state = Require-State
    Invoke-UvPython @(
        'tools/boundary/ja/train_candidate_island_scorer_v11.py',
        '--dataset-manifest', (Join-Path $state.features_dir 'training_windows.jsonl'),
        '--feature-cache-gate', (Join-Path $state.features_dir 'feature_cache_gate.json'),
        '--output-dir', [string]$state.smoke_dir,
        '--variant', 'baseline',
        '--capacity-profile', 'full_p2048_h256',
        '--device', 'cuda',
        '--smoke',
        '--seed', '117',
        '--epochs', '1',
        '--max-steps', '60',
        '--max-padded-frames', [string]$state.max_padded_frames,
        '--source-cache-size', '4',
        '--learning-rate', '2e-4',
        '--weight-decay', '1e-4',
        '--log-every', '10'
    )
}

function Invoke-Full {
    $state = Require-State
    if (-not (Test-Path -LiteralPath (Join-Path $state.smoke_dir 'summary.json'))) {
        throw 'CUDA smoke summary is missing; full training is refused.'
    }
    Invoke-UvPython @(
        'tools/boundary/ja/train_candidate_island_scorer_v11.py',
        '--dataset-manifest', (Join-Path $state.features_dir 'training_windows.jsonl'),
        '--feature-cache-gate', (Join-Path $state.features_dir 'feature_cache_gate.json'),
        '--output-dir', [string]$state.full_dir,
        '--variant', 'baseline',
        '--capacity-profile', 'full_p2048_h256',
        '--device', 'cuda',
        '--seed', '117',
        '--epochs', '20',
        '--max-padded-frames', [string]$state.max_padded_frames,
        '--source-cache-size', '4',
        '--learning-rate', '2e-4',
        '--weight-decay', '1e-4',
        '--log-every', '50',
        '--eval-every-epochs', '1',
        '--early-stopping-patience', '3',
        '--early-stopping-min-delta', '1e-4'
    )
}

function Invoke-Gate {
    $state = Require-State
    $training = Read-Json -Path (Join-Path $state.full_dir 'summary.json')
    if (-not $training.checkpoint) {
        throw 'Full training summary has no checkpoint.'
    }
    Invoke-UvPython @(
        'tools/audits/score_candidate_island_scorer_v11_checkpoint.py',
        '--checkpoint', [string]$training.checkpoint,
        '--canonical-sources', [string]$state.canonical_sources,
        '--raw-feature-manifest', [string]$state.raw_feature_manifest,
        '--output-dir', [string]$state.gate_dir,
        '--partition', 'val',
        '--partition', 'test',
        '--device', 'cuda',
        '--max-padded-frames', [string]$state.max_padded_frames,
        '--tolerance-frames', '15',
        '--long-residual-frames', '400'
    )
    $auditName = (Split-Path -Leaf $RunRoot) + '-full-source-gate'
    $auditOutput = Join-Path $ProjectRoot (Join-Path 'agents/audits' $auditName)
    Invoke-UvPython @(
        'tools/audits/generate_candidate_island_scorer_v11_prediction_audit_html.py',
        '--source-predictions', (Join-Path $state.gate_dir 'source_predictions.jsonl'),
        '--output-dir', $auditOutput
    )
    Write-Host "Audit URL: http://127.0.0.1:8080/agents/audits/$auditName/"
}

switch ($Stage) {
    'prepare' { Invoke-Prepare }
    'extract' { Invoke-Extract }
    'features' { Invoke-Features }
    'smoke' { Invoke-Smoke }
    'full' { Invoke-Full }
    'gate' { Invoke-Gate }
    'all' {
        Invoke-Prepare
        Invoke-Extract
        Invoke-Features
        Invoke-Smoke
        Invoke-Full
        Invoke-Gate
    }
}

Write-Host "Scorer v11 retrain stage '$Stage' completed. Run root: $RunRoot"
