# Windows release packaging

Build from the repository root after creating the project virtual environment
with `uv venv` and installing dependencies:

```powershell
.\packaging\build_windows.ps1 -Clean
```

The build creates `dist/JAVTrans/JAVTrans.exe` as an onedir PyInstaller package.
It bundles:

- the Python runtime and installed Python dependencies from the active uv-managed environment
- `ffmpeg.exe` and `ffprobe.exe` from `PATH`, or from `-FfmpegExe` / `-FfprobeExe`
- `icon.ico` for the executable and pywebview window
- `icon.png` / `icon.ico` for the in-app header and favicon
- the default ASR model `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`
- default workflow auxiliary models:
  - forced aligner `Qwen/Qwen3-ForcedAligner-0.6B`
  - learned Boundary Refiner `src/boundary/checkpoints/boundary_refiner.pt`

The PyInstaller spec treats `src/boundary/checkpoints/` as a required data
directory. A build should fail if the default Boundary Refiner checkpoint is
missing, because normal inference does not regenerate it.

It does not bundle Microsoft Edge WebView2. Users still need the WebView2 runtime,
which is already present on most supported Windows systems. If the app window
does not open, install Evergreen Runtime from Microsoft's official WebView2
download page: https://developer.microsoft.com/en-us/microsoft-edge/webview2/.
Use Evergreen Bootstrapper for online installs, or Evergreen Standalone Installer
x64 for offline installs.

The release folder is expected to be several GB because it contains PyTorch/CUDA
runtime files plus the default ASR model. GitHub currently requires every
uploaded release asset to be under 2 GiB, so publish this as split archives or
host the large model/runtime bundle externally and link it from the release
notes. See GitHub Docs: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas

At runtime, writable files are created next to `JAVTrans.exe`:

- `.env` for persisted settings
- `models/` for user-downloaded non-default models
- `tmp/jobs/` for job state and resumable task files
- `tmp/log/` for `.run.log` diagnostics that users can attach to bug reports

To create split archives with 7-Zip:

```powershell
.\packaging\archive_release.ps1
```

The default output is `dist/release-assets/JAVTrans-windows-x64.7z.001`,
`.002`, etc. Users extract starting from the `.001` file.
The script uses all logical CPU threads by default; override with `-Threads <n>`
if you need to keep the machine responsive while compressing.

Only the default ASR model, forced aligner, runtime assets, and the small
`src/boundary/checkpoints/boundary_refiner.pt` are bundled. Other ASR
models remain download-on-demand into the executable folder's `models/`
directory.

Training-only Boundary Refiner artifacts are deliberately excluded from release
packages: CUDA feature caches, synthetic WAVs, sequence JSONL files, and
`datasets/train/...` outputs are all regenerable research data. New users only
need the bundled `boundary_refiner.pt` plus the Hugging Face ASR / aligner models
above. Do not restore old `src/vad` checkpoint paths; if the Boundary Refiner
checkpoint grows too large for source distribution, publish it as a GitHub
Release or Hugging Face artifact instead.
