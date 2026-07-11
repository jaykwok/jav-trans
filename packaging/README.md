# Windows release packaging

Build from the repository root after creating the project virtual environment
with `uv venv` and installing dependencies:

```powershell
.\packaging\build_windows.ps1 -Clean
```

The build creates `dist/jav-trans/jav-trans.exe` as an onedir PyInstaller package.
It bundles:

- the Python runtime and installed Python dependencies from the active uv-managed environment
- `ffmpeg.exe`, `ffprobe.exe`, and FFmpeg runtime DLLs from the Shared build on
  `PATH`, or from `-FfmpegExe` / `-FfprobeExe`
- `src/assets/images/icon.png` for the in-app header, drop zone image, and PNG favicon
- `src/assets/images/icon.ico` for the pywebview native window icon and packaged executable icon
- all repo-bound small models under `src/checkpoints/<repo-tag>/`
- the bundled Hugging Face inference model directories:
  - `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`
  - `jaykwok/Qwen3-ASR-0.6B-JA-Anime-Galgame-hf`

The build script prepares those two Hugging Face models before running
PyInstaller. Training-only files such as `optimizer.pt`, scheduler state,
trainer state, RNG state, and `training_args.bin` are excluded from the package
even if they exist in the local `models/` directories.

On Windows, install the FFmpeg Shared package before building:

```powershell
winget uninstall --id Gyan.FFmpeg --exact
winget install --id Gyan.FFmpeg.Shared --exact
```

The directory selected for `ffmpeg.exe` must also contain `avcodec-*.dll`,
`avformat-*.dll`, and `avutil-*.dll`; TorchCodec cannot use the static-only
`Gyan.FFmpeg` package. If multiple FFmpeg installations are present, pass the
Shared executables explicitly with `-FfmpegExe` and `-FfprobeExe`.

For a small development build only, pass `-SkipModels`. That skips model
preparation and leaves the Hugging Face model directories out of the PyInstaller
package. Do not use `-SkipModels` for user-facing Windows builds.

The PyInstaller spec treats `src/checkpoints/` as required data. It contains one
directory per supported ASR repo and includes both active models and explicit
offline replay checkpoints. Normal inference does not regenerate these files.

It does not bundle Microsoft Edge WebView2. Users still need the WebView2
runtime, which is already present on most supported Windows systems. If the app
window does not open, install Evergreen Runtime from Microsoft's official
WebView2 download page: https://developer.microsoft.com/en-us/microsoft-edge/webview2/.
Use Evergreen Bootstrapper for online installs, or Evergreen Standalone
Installer x64 for offline installs.

The Windows package bundles the CUDA-enabled PyTorch runtime DLLs, but it does
not bundle the NVIDIA display driver. The user's installed driver must support
the CUDA runtime reported by the bundled PyTorch build. On startup/model
selection the Web UI runs a short CUDA probe in a child process; if the driver is
too old or CUDA cannot initialize, it shows a user-facing prompt to update the
NVIDIA driver before running ASR.

At runtime, writable files are created next to `jav-trans.exe`:

- `.env` for persisted settings
- `models/` for user-downloaded or user-replaced models
- `tmp/jobs/` for job state and resumable task files
- `tmp/log/` for `.run.log` diagnostics that users can attach to bug reports

To create one local 7-Zip archive:

```powershell
.\packaging\archive_release.ps1
```

The default output is `dist/release-assets/jav-trans-windows-x64.7z`. The archive
script creates a single `.7z` file and no split volumes. Publish this large
Windows bundle through external storage such as a netdisk; GitHub Releases are
expected to publish source code and release notes only.

Training-only Mamba artifacts are deliberately excluded from release
packages: CUDA feature caches, synthetic WAVs, sequence JSONL files, and
`datasets/train/...` outputs are all regenerable research data. New users only
need the bundled repo-tagged Boundary Refiner and SpeechBoundary-JA scorer
plus the bundled Hugging Face inference models above. Do not
restore old `src/vad` checkpoint paths; if Mamba checkpoints grow too large for
source distribution, publish them as GitHub Release or Hugging Face artifacts
instead.
