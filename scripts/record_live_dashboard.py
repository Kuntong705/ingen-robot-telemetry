"""Screen-record the live Week 2 dashboard (Windows, FFmpeg gdigrab window capture).

Starts ``scripts/w02_live_dashboard.py`` (which computes everything in this run),
waits for its window, records only that window until the program closes it,
then encodes ``media/W02_live_dashboard_recording.mp4``. Existing videos are
never overwritten.

    python scripts/record_live_dashboard.py
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "W02 live dashboard"
PROTECTED = {"W02_demo.mp4", "W02_screen_recording.mp4"}
CREATE_NO_WINDOW = 0x08000000


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_demo_video import find_ffmpeg

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "media" / "W02_live_dashboard_recording.mp4")
    parser.add_argument("--cap-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.out.name in PROTECTED:
        raise SystemExit(f"refusing to overwrite {args.out.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONWARNINGS": "ignore", "MLFLOW_DISABLE_AGENT_HINT": "1"}
    app = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "w02_live_dashboard.py")], cwd=ROOT, env=env,
                           creationflags=CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True)
    user32 = ctypes.windll.user32
    hwnd = 0
    for _ in range(1800):  # imports happen before the window opens (up to ~90 s)
        hwnd = user32.FindWindowW(None, TITLE)
        if hwnd or app.poll() is not None:
            break
        time.sleep(0.05)
    if not hwnd:
        raise SystemExit(f"dashboard window did not appear: {app.stderr.read()[-800:] if app.stderr else ''}")
    # gdigrab fixes the frame size when it starts: wait until Tk has laid the window out at full size
    rect = ctypes.wintypes.RECT()
    for _ in range(100):
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        if rect.right >= 1200 and rect.bottom >= 680:
            break
        time.sleep(0.05)
    time.sleep(0.3)
    user32.SetForegroundWindow(hwnd)

    ffmpeg = find_ffmpeg()
    raw = args.out.with_name(args.out.stem + "_raw.mkv")
    rec = subprocess.Popen([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "gdigrab", "-framerate", "20",
                            "-draw_mouse", "0", "-i", f"title={TITLE}", "-t", str(args.cap_seconds),
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "14", str(raw)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while app.poll() is None and rec.poll() is None:
        time.sleep(0.1)
    if rec.poll() is None:
        assert rec.stdin is not None
        rec.stdin.write("q")
        rec.stdin.flush()
    rec.wait(timeout=30)
    err = app.stderr.read() if app.stderr else ""
    if app.wait(timeout=30) != 0:
        raise SystemExit(f"dashboard failed: {err[-1500:]}")
    probe = subprocess.run([str(Path(ffmpeg).with_name("ffprobe.exe")), "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(raw)], capture_output=True, text=True, check=True)
    keep = max(1.0, float(probe.stdout.strip()) - 0.5)  # drop the window-closing frames
    final = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-t", f"{keep:.2f}",
                            # the dashboard renders at 96 DPI: keep exactly its 1280x720 surface
                            "-vf", "crop=1280:720:0:0", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-metadata", "comment=Real screen recording of scripts/w02_live_dashboard.py (window only)",
                            str(args.out)], capture_output=True, text=True, check=False)
    if final.returncode != 0:
        raise SystemExit(f"encode failed: {final.stderr[-500:]}")
    raw.unlink()
    print(args.out)


if __name__ == "__main__":
    main()
