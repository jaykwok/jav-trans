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
- `src/assets/images/icon.png` for the in-app header, drop zone image, and PNG favicon
- `src/assets/images/icon.ico` for the pywebview native window icon and packaged executable icon
- the small Mamba Boundary Refiner checkpoint at `src/boundary/checkpoints/boundary_refiner.pt`
- the bundled Hugging Face inference model directories:
  - `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame`
  - `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame`
  - `Qwen/Qwen3-ForcedAligner-0.6B`

The build script prepares those three Hugging Face models before running
PyInstaller. Training-only files such as `optimizer.pt`, scheduler state,
trainer state, RNG state, and `training_args.bin` are excluded from the package
even if they exist in the local `models/` directories.

For a small development build only, pass `-SkipModels`. That skips model
preparation and leaves the Hugging Face model directories out of the PyInstaller
package. Do not use `-SkipModels` for user-facing Windows builds.

The PyInstaller spec treats `src/boundary/checkpoints/` as a required data
directory. A build should fail if the default Boundary Refiner checkpoint is
missing, because normal inference does not regenerate it.

It does not bundle Microsoft Edge WebView2. Users still need the WebView2
runtime, which is already present on most supported Windows systems. If the app
window does not open, install Evergreen Runtime from Microsoft's official
WebView2 download page: https://developer.microsoft.com/en-us/microsoft-edge/webview2/.
Use Evergreen Bootstrapper for online installs, or Evergreen Standalone
Installer x64 for offline installs.

At runtime, writable files are created next to `JAVTrans.exe`:

- `.env` for persisted settings
- `models/` for user-downloaded or user-replaced models
- `tmp/jobs/` for job state and resumable task files
- `tmp/log/` for `.run.log` diagnostics that users can attach to bug reports

To create one local 7-Zip archive:

```powershell
.\packaging\archive_release.ps1
```

The default output is `dist/release-assets/JAVTrans-windows-x64.7z`. The archive
script creates a single `.7z` file and no split volumes. Publish this large
Windows bundle through external storage such as a netdisk; GitHub Releases are
expected to publish source code and release notes only.

Training-only Boundary Refiner artifacts are deliberately excluded from release
packages: CUDA feature caches, synthetic WAVs, sequence JSONL files, and
`datasets/train/...` outputs are all regenerable research data. New users only
need the bundled `boundary_refiner.pt` plus the bundled Hugging Face inference
models above. Do not restore old `src/vad` checkpoint paths; if the Boundary
Refiner checkpoint grows too large for source distribution, publish it as a
GitHub Release or Hugging Face artifact instead.
