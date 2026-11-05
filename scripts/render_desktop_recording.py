"""Post-process the real desktop recording made by desktop_demo.py.

Edits to the captured footage (all disclosed; nothing is added, re-ordered or re-drawn):
  * privacy blur: PyCharm's run console path lines and the GitHub username / avatars /
    Edge profile picture;
  * the wait while the worker process starts is sped up by the smallest whole-number
    factor (>= 3) that keeps the video <= MAX_TOTAL_S, with an on-screen "Nx speed" label;
  * the static GitHub page-load wait after the click is shortened: 0.6 s after the click is
    kept, then the video continues from the frame where the run page has loaded.

Event times come from the log written during the recording and are mapped onto the video
timeline with the measured offset (video duration - last event time); the page-load moment is
detected from the frames themselves.

    python scripts/render_desktop_recording.py [RAW] [OUT]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from desktop_demo import ffmpeg_exe

PROTECTED = {"W02_demo.mp4", "W02_screen_recording.mp4", "W02_live_dashboard_recording.mp4",
             "W02_desktop_recording.mp4"}
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
MIN_SPEEDUP = 3
MAX_TOTAL_S = 59.5
KEEP_AFTER_CLICK_S = 0.6
SUCCESS_HOLD_S = 4.0  # the loaded Success page stays on screen at least this long


def event_time(events: list[list], prefix: str) -> float:
    return next(float(t) for t, text in events if text.startswith(prefix))


def duration(path: Path) -> float:
    probe = Path(ffmpeg_exe()).with_name("ffprobe.exe")
    out = subprocess.run([str(probe), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def page_loaded_at(raw: Path, t_click: float) -> float:
    """First frame after the click where the page heading area differs clearly from the run list."""
    x, y, w, h, fps = 30, 240, 680, 80, 20
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{t_click:.2f}", "-t", "12", "-i", str(raw),
           "-vf", f"fps={fps},crop={w}:{h}:{x}:{y},format=gray", "-f", "rawvideo", "-"]
    frames = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, h, w)
    diff = np.abs(frames.astype(np.int16) - frames[0]).mean(axis=(1, 2))
    hit = int(np.argmax(diff > 12))
    if diff[hit] <= 12:
        raise SystemExit("run page never loaded in the recording")
    return t_click + hit / fps


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "media" / "W02_desktop_raw_final.mkv"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "media" / "W02_desktop_recording_final.mp4"
    if out.name in PROTECTED:
        raise SystemExit(f"refusing to overwrite {out.name}")
    events = json.loads(raw.with_suffix(".events.json").read_text(encoding="utf-8"))
    offset = duration(raw) - float(events[-1][0])  # video time = event time + offset (ffmpeg start-up)

    def vt(prefix: str) -> float:
        return event_time(events, prefix) + offset

    t_first, t_run, t_dash = vt("Ctrl+Shift+N: open models"), vt("Ctrl+Shift+F10"), vt("dashboard window open")
    t_closed, t_edge, t_click = vt("dashboard finished"), vt("switch to Edge"), vt("click:")
    t_loaded = page_loaded_at(raw, t_click)

    start, fast_a, fast_b = max(0.0, t_first - 1.0), t_run + 1.0, t_dash - 0.5
    cut_a, cut_b, end = t_click + KEEP_AFTER_CLICK_S, t_loaded - 0.1, t_loaded + SUCCESS_HOLD_S + 0.1
    normal = (fast_a - start) + (cut_a - fast_b) + (end - cut_b)
    speedup = MIN_SPEEDUP
    while normal + (fast_b - fast_a) / speedup > MAX_TOTAL_S:
        speedup += 1

    # (x, y, w, h, t_from, t_to) on the video timeline, 1920x1080 physical pixels
    pc_band = (100, 700, 1770, 92)  # path lines only; "process finished with exit code 0" stays readable
    regions = [
        (*pc_band, 0, t_dash + 1.0),             # run console (before the dashboard covers it)
        (*pc_band, t_closed - 0.5, t_edge + 1.5),
        (100, 700, 72, 92, 0, t_edge + 1.5),     # console slivers left/right of the dashboard window
        (1766, 700, 104, 92, 0, t_edge + 1.5),
        # username in the URL bar; wide enough for both layouts (with/without Edge's Forward button)
        (222, 58, 258, 34, t_edge - 0.5, 1e9),
        (0, 1036, 720, 44, t_edge - 0.5, 1e9),   # status bar: link URL shown while hovering the run
        (130, 124, 110, 34, t_edge - 0.5, 1e9),  # breadcrumb username
        (1830, 112, 56, 56, t_edge - 0.5, 1e9),  # GitHub avatar
        (1734, 54, 50, 44, t_edge - 0.5, 1e9),   # Edge profile picture
        (774, 506, 96, 30, t_edge - 0.5, 1e9),   # "pushed by <user>" on the run list
        (474, 410, 142, 36, t_loaded - 0.3, 1e9),  # "<avatar> <user> pushed" on the run page
    ]
    parts, last = [], "0:v"
    for i, (x, y, w, h, a, b) in enumerate(regions):
        parts.append(f"[0:v]crop={w}:{h}:{x}:{y},boxblur=12:3[b{i}]")
        parts.append(f"[{last}][b{i}]overlay={x}:{y}:enable='between(t,{a:.2f},{b:.2f})'[v{i}]")
        last = f"v{i}"
    label = (f"drawtext=fontfile='{FONT}':text='{speedup}x speed  (worker process starting)':fontsize=34:"
             "fontcolor=white:box=1:boxcolor=0x0b1220@0.85:boxborderw=14:x=w-tw-60:y=h-th-90")
    parts.append(f"[{last}]split=4[s1][s2][s3][s4]")
    parts.append(f"[s1]trim={start:.2f}:{fast_a:.2f},setpts=PTS-STARTPTS[p1]")
    parts.append(f"[s2]trim={fast_a:.2f}:{fast_b:.2f},setpts=(PTS-STARTPTS)/{speedup},{label}[p2]")
    parts.append(f"[s3]trim={fast_b:.2f}:{cut_a:.2f},setpts=PTS-STARTPTS[p3]")
    parts.append(f"[s4]trim={cut_b:.2f}:{end:.2f},setpts=PTS-STARTPTS[p4]")
    parts.append("[p1][p2][p3][p4]concat=n=4:v=1:a=0,fps=20,format=yuv420p[out]")
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-filter_complex",
           ";".join(parts), "-map", "[out]", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    total = normal + (fast_b - fast_a) / speedup
    edit = {"offset_s": round(offset, 2), "start": start, "speedup": speedup, "fast": [fast_a, fast_b],
            "page_load_cut": [cut_a, cut_b], "page_loaded_at": t_loaded, "end": end, "expected_s": round(total, 2)}
    out.with_suffix(".edit.json").write_text(json.dumps(edit, indent=1), encoding="utf-8")
    print(f"wrote {out}  expected {total:.1f} s; {speedup}x on {fast_a:.1f}-{fast_b:.1f} s; "
          f"page-load wait {cut_a:.1f}-{cut_b:.1f} s removed (loaded at {t_loaded:.1f} s)")


if __name__ == "__main__":
    main()
