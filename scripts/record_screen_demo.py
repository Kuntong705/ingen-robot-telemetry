"""Screen-record one real run of ``scripts/w02_demo.py`` (Windows, FFmpeg gdigrab).

Opens a dedicated console window titled "W02 live demo", places it at the
top-left of the screen, and records only that window (gdigrab title capture)
while the demo program runs in it. Output: ``media/W02_screen_recording.mp4``.
This is a real screen recording of the running program. It does not overwrite
``media/W02_demo.mp4`` (the rendered video) or its log.

    python scripts/record_screen_demo.py
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "W02 live demo"
CREATE_NEW_CONSOLE = 0x00000010


def find_ffmpeg() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_demo_video import find_ffmpeg as _find

    return _find()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "media" / "W02_screen_recording.mp4")
    parser.add_argument("--seconds", type=int, default=60, help="hard cap; recording stops when the demo ends")
    args = parser.parse_args()
    if args.out.resolve() == (ROOT / "media" / "W02_demo.mp4").resolve():
        raise SystemExit("refusing to overwrite the rendered demo video")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    user32 = ctypes.windll.user32
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixels, matching gdigrab
    ps1 = ROOT / "scripts" / "screen_demo_window.ps1"
    window = subprocess.Popen(
        ["conhost.exe", "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
        creationflags=CREATE_NEW_CONSOLE, cwd=ROOT,
    )
    hwnd = 0
    for _ in range(100):
        hwnd = user32.FindWindowW(None, TITLE)
        if hwnd:
            break
        time.sleep(0.1)
    if not hwnd:
        window.kill()
        raise SystemExit("demo window did not appear")
    # top-left, away from the notification area; even dimensions for H.264
    user32.ShowWindow(hwnd, 9)
    user32.MoveWindow(hwnd, 40, 40, 1400, 820, True)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)

    raw = args.out.with_name(args.out.stem + "_raw.mkv")
    ffmpeg = [
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "gdigrab", "-framerate", "15", "-draw_mouse", "0", "-i", f"title={TITLE}",
        "-t", str(args.seconds), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(raw),
    ]
    # record until the demo window closes (or the hard cap), with little CPU overhead
    rec = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while window.poll() is None and rec.poll() is None:
        time.sleep(0.2)
    if rec.poll() is None:
        assert rec.stdin is not None
        rec.stdin.write("q")
        rec.stdin.flush()
    rec.wait(timeout=30)
    window.wait(timeout=60)
    # final encode: even dimensions, web-friendly MP4
    final = subprocess.run([
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-metadata", "comment=Real screen recording of scripts/w02_demo.py (window capture only)",
        str(args.out),
    ], capture_output=True, text=True, check=False)
    if final.returncode != 0:
        raise SystemExit(f"ffmpeg encode failed: {final.stderr[-500:]}")
    raw.unlink()
    print(args.out)


if __name__ == "__main__":
    main()
