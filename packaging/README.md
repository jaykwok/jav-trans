# Windows release packaging

There are two builds. The setup build is what users get; the full bundle is the
offline fallback.

## Setup build (default for releases)

```powershell
.\packaging\build_setup.ps1 -Clean
```

Produces `dist/release-assets/release-<timestamp>-<id>/jav-trans-windows-x64.zip`
(~100 MB), with `release-manifest.json` and a `.zip.sha256` file. Its payload
is `jav-trans.exe`, `bin/` (FFmpeg Shared), `src/`, `launcher.py`,
`pyproject.toml`, `uv.lock`, and `README.txt`. FFmpeg is most of the size.

The exe is named `jav-trans.exe`, not `jav-trans-setup.exe`: installing is only
what the first run does, and it is the launcher on every run after that. Both
specs therefore build a target named `jav-trans`, so this build stages its
payload under `dist/setup-payload/jav-trans` and passes its own `--workpath`
(`build/jav-trans-setup`) - one PyInstaller cache shared by two different
Analysis inputs is a stale-build trap.

Python, PyTorch, the ASR weights, and the CTC head are not in the archive. The
installer obtains Python as needed and installs the locked dependencies on the
user's machine; the first ASR task downloads the model and alignment head.
This keeps the release around 100 MB. The console stays open during `uv sync`
so the user can see the download rate and decide whether they need a proxy;
`bootstrap.py` measures against the real torch wheel named in `uv.lock` and
reports the estimated time rather than applying a threshold.

FFmpeg travels with the archive because TorchCodec loads its shared DLLs at
import time and uv cannot install them. `launcher.py` finds them at `bin/`.

uv is not bundled. The first run uses one already on `PATH`, and otherwise
downloads the Windows wheel from PyPI into `bin/` - the same host the dependency
install needs a moment later, so shipping 50-80 MB of uv would buy no
reachability that the rest of the install does not already require. `-BundleUv`
includes it anyway, for building an archive that has to travel to a network
where PyPI itself is blocked.

Options: `-SkipArchive` stops after the payload directory, `-UvExe` /
`-FfmpegExe` / `-FfprobeExe` override tool discovery, `-ArchiveName` renames the
zip. zip rather than the `.7z` below because this archive is small enough that
the ratio does not matter and Windows opens zip with nothing installed.

Keep `build_setup.ps1` ASCII-only: Windows PowerShell 5.1 reads a BOM-less
`.ps1` as the system code page, so a Chinese string literal in it ships as
mojibake.

## Full bundle (offline)

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
- `models/ctc_aligner.pt`, the CTC alignment head
- the bundled Hugging Face inference model directory
  `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`

The build script prepares that Hugging Face model before running
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

The CTC alignment head is downloaded at build time from the same Hugging Face
repo as the ASR weights, at the commit sha pinned in
`DEFAULT_SETTINGS["ASR_ALIGNMENT_HEAD_PATH"]`, and placed at `models/ctc_aligner.pt`
inside the package. The spec reads that default rather than hardcoding the sha,
so the head the build ships is always the head a source checkout would download.
The packaged app prefers this bundled copy over the Hub, so a first run without
network still produces real word-level timing instead of falling back to
proportional timestamps. `-SkipModels` skips it along with the ASR models.

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
- `tmp/web/` for persisted job state
- `tmp/jobs/` for resumable task files
- `tmp/outputs/` for immutable generations of completed outputs; these are not
  disposable caches and remain after deleting a job
- `tmp/log/` for `.run.log` diagnostics that users can attach to bug reports

To create one local 7-Zip archive:

```powershell
.\packaging\archive_release.ps1
```

The default output is
`dist/release-assets/release-<timestamp>-<id>/jav-trans-windows-x64.7z`, with
`.7z.sha256` and `release-manifest.json` beside it. The archive script creates
a single `.7z` file and no split volumes. ZIP setup archives use the same
publication path: freeze the build manifest, verify the source, compress in a
private directory, extract and verify the actual archived bytes, then rename
the complete directory into place. A failed check publishes nothing. Existing
release directories are never overwritten. Publish this large offline bundle
through external storage such as a netdisk. GitHub Releases use the small setup
ZIP described above, alongside the release notes and source archives.

## Reproducibility: pinned weights and a hash manifest

The full bundle's ASR model is downloaded at a pinned commit, recorded in
`packaging/model-pins.json`. `build_windows.ps1` refuses to build without a valid
full commit SHA,
because a release that says "the 1.7B ASR model" and takes whatever the branch
points at that afternoon is not the same package twice. Refresh the pin
deliberately, as its own step:

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python packaging/prepare_default_model.py --pin
```

Pinning queries public Hub metadata; it does not download model weights.
Preparing a full bundle downloads into `models/.pinned/<repo>/<commit>/`,
separately from the application's ordinary model cache. A `model-revision.json`
receipt records the repository, SHA, sizes and SHA256 hashes of every inference
file. Preparation, spec data collection and the final payload all verify this
receipt. A modified or incomplete pinned directory is refused; an unrelated
cached model cannot satisfy the pin. There is no unpinned full-build option.
The CTC alignment head is pinned by SHA through
`DEFAULT_SETTINGS["ASR_ALIGNMENT_HEAD_PATH"]`, and the packaged app loads it with
`weights_only=True` - a head fetched from the Hub is not a file the user wrote.

After PyInstaller succeeds, the build records
`dist/release-assets/release-manifest.json`: SHA256 of every payload file,
including code, bytecode, configuration, templates, model weights and tool
binaries, plus `uv.lock`, `pyproject.toml`, model pins and the source commit's
dirty status. The setup build writes `setup-manifest.json` before archiving,
even with `-SkipArchive`. Each published generation contains its own frozen
copy named `release-manifest.json`.

The setup ZIP hashes the files it ships. Its first-run downloads are outside
that inventory, so this does not promise an identical installed environment or
fixed ASR weights for setup users. Use the full bundle when the model payload
must be fixed. Neither path promises bit-for-bit identical PyInstaller archives.

To check an extracted payload from the corresponding source checkout:

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python packaging/record_release_manifest.py --dist dist/jav-trans --verify dist/release-assets/release-<timestamp>-<id>/release-manifest.json
```

Paths in the manifest are relative to the payload and to the repository, so it
carries nothing about the machine that built it.

Training-only artifacts, CUDA feature caches, synthetic WAVs and sequence JSONL
files are excluded. The full bundle ships the default ASR model and CTC head
listed above; retired Mamba and VAD checkpoints are not release inputs.
