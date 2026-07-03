from __future__ import annotations

import os
import subprocess


def no_window_subprocess_kwargs() -> dict[str, int]:
    """Hide console windows for child console programs from a windowed app.

    Windows console executables such as ffmpeg.exe, ffprobe.exe and
    nvidia-smi.exe can create a transient black console window when launched
    from a PyInstaller windowed executable. CREATE_NO_WINDOW is ignored on
    non-Windows platforms by not passing it there.
    """
    if os.name != "nt":
        return {}
    flag = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
    return {"creationflags": flag} if flag else {}
