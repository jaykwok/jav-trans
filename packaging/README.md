# Windows release packaging

Build from the repository root with the project virtual environment:

```powershell
.\packaging\build_windows.ps1 -Clean
```

The build creates `dist/JAVTrans/JAVTrans.exe` as an onedir PyInstaller package.
It bundles:

- the Python runtime and installed Python dependencies from `.venv`
- `ffmpeg.exe` and `ffprobe.exe` from `PATH`, or from `-FfmpegExe` / `-FfprobeExe`
- `icon.ico` for the executable and pywebview window
- `icon.png` / `icon.ico` for the in-app header and favicon
- the default ASR model `efwkjn/whisper-ja-anime-v0.3`
- default workflow auxiliary models:
  - WhisperSeg VAD `TransWithAI/Whisper-Vad-EncDec-ASMR-onnx`
  - WhisperSeg feature extractor `openai/whisper-base`
  - forced aligner `Qwen/Qwen3-ForcedAligner-0.6B`

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
- `temp/web/` for job state and resumable task files
- `temp/log/` for `.run.log` diagnostics that users can attach to bug reports

To create split archives with 7-Zip:

```powershell
.\packaging\archive_release.ps1
```

The default output is `dist/release-assets/JAVTrans-windows-x64.7z.001`,
`.002`, etc. Users extract starting from the `.001` file.
The script uses all logical CPU threads by default; override with `-Threads <n>`
if you need to keep the machine responsive while compressing.

Only the default ASR model and the default workflow auxiliary models are bundled.
Other ASR models remain download-on-demand into the executable folder's `models/`
directory.
