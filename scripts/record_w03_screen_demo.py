"""Screen-record one real run of ``scripts/w03_demo.py`` (Windows, FFmpeg gdigrab) as a 30-60 s MP4.

Opens a dedicated console window titled "W03 live demo" and records only that
window (gdigrab title capture, no desktop, no mouse) while the demo program runs
in it. The local LLM needs minutes on this CPU, so two waiting segments are
played faster in the final cut: the Python start-up, and the LLM loading and
generating. Each sped-up segment carries an on-screen label with its speed
factor and its real duration; everything else is real time. No frame is
synthesised: the MP4 is the recorded window, re-timed.

Output: ``media/W03_demo.mp4`` (git-ignored) plus ``media/W03_demo.edit.json``
with the cut list and ``media/w03_demo_events.json`` from the demo program.

    python scripts/record_w03_screen_demo.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLE = "W03 live demo"
CREATE_NEW_CONSOLE = 0x00000010
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
MAX_TOTAL_S = 59.0
MIN_TOTAL_S = 30.0
PROTECTED = {"W02_demo.mp4", "W02_screen_recording.mp4", "W02_live_dashboard_recording.mp4",
             "W02_desktop_recording.mp4", "W02_desktop_recording_final.mp4"}


def find_ffmpeg() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_demo_video import find_ffmpeg as _find

    return str(_find())


def duration(path: Path) -> float:
    probe = Path(find_ffmpeg()).with_name("ffprobe.exe")
    out = subprocess.run([str(probe), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def record(raw: Path, alert: str, cap_s: int) -> tuple[float, float]:
    """Run the demo in its own console window and record that window. Returns wall-clock start/stop."""
    user32 = ctypes.windll.user32
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixels, matching gdigrab
    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    w, h = min(1600, sw - 80) // 2 * 2, min(1000, sh - 80) // 2 * 2
    ps1 = ROOT / "scripts" / "w03_screen_demo_window.ps1"
    env = dict(os.environ, W03_DEMO_FONT="18", W03_DEMO_RECT=f"40,30,{w},{h}")
    window = subprocess.Popen(
        ["conhost.exe", "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), "-Alert", alert],
        creationflags=CREATE_NEW_CONSOLE, cwd=ROOT, env=env,
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
    user32.ShowWindow(hwnd, 9)
    user32.MoveWindow(hwnd, 40, 30, w, h, True)
    user32.SetForegroundWindow(hwnd)
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "gdigrab", "-framerate", "15",
           "-draw_mouse", "0", "-i", f"title={TITLE}", "-t", str(cap_s), "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "14", str(raw)]
    t_start = time.time()
    rec = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while window.poll() is None and rec.poll() is None:
        time.sleep(0.2)
    t_stop = time.time()
    if rec.poll() is None:
        assert rec.stdin is not None
        rec.stdin.write("q")
        rec.stdin.flush()
    rec.wait(timeout=60)
    window.wait(timeout=60)
    return t_start, t_stop


def render(raw: Path, out: Path, events_path: Path, t_start: float, t_stop: float) -> dict[str, object]:
    log = json.loads(events_path.read_text(encoding="utf-8"))
    ev = {name: t for name, t in log["events"]}
    d = duration(raw)
    latency = (t_stop - t_start) - d  # ffmpeg start-up: video time = wall - t_start - latency

    def vt(name: str) -> float:
        return ev[name] - t_start - latency

    hold = 1.5  # banner shown at normal speed
    a1, b1 = min(hold, vt("first_output")), max(min(hold, vt("first_output")), vt("first_output") - 0.2)
    a2, b2 = vt("llm_load_start") + 1.0, vt("llm_generate_end")
    normal = a1 + (a2 - b1) + (d - b2)
    fast = (b1 - a1) + (b2 - a2)
    if normal > MAX_TOTAL_S:
        raise SystemExit(f"real-time parts alone last {normal:.1f} s; cannot fit 60 s")
    speed = 1
    while normal + fast / speed > MAX_TOTAL_S:
        speed += 1

    def label(text: str) -> str:
        return (f"drawtext=fontfile='{FONT}':text='{text}':fontsize=30:fontcolor=white:box=1:"
                "boxcolor=0x0b1220@0.85:boxborderw=12:x=w-tw-30:y=24")

    l1 = label(f"{speed}x speed - Python start-up (real time {b1 - a1:.0f} s)")
    l2 = label(f"{speed}x speed - local LLM loading and generating on CPU (real time {b2 - a2:.0f} s)")
    parts = [
        "[0:v]scale=trunc(iw/2)*2:trunc(ih/2)*2,split=5[s0][s1][s2][s3][s4]",
        f"[s0]trim=0:{a1:.3f},setpts=PTS-STARTPTS[p0]",
        f"[s1]trim={a1:.3f}:{b1:.3f},setpts=(PTS-STARTPTS)/{speed},{l1}[p1]",
        f"[s2]trim={b1:.3f}:{a2:.3f},setpts=PTS-STARTPTS[p2]",
        f"[s3]trim={a2:.3f}:{b2:.3f},setpts=(PTS-STARTPTS)/{speed},{l2}[p3]",
        f"[s4]trim={b2:.3f}:{d:.3f},setpts=PTS-STARTPTS[p4]",
        "[p0][p1][p2][p3][p4]concat=n=5:v=1:a=0,fps=15,format=yuv420p[out]",
    ]
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw), "-filter_complex", ";".join(parts),
           "-map", "[out]", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-movflags", "+faststart",
           "-metadata", (f"comment=Screen recording of scripts/w03_demo.py (window capture only); {speed}x on two "
                         "labelled waiting segments"), str(out)]
    subprocess.run(cmd, check=True)
    return {"raw_duration_s": round(d, 2), "ffmpeg_latency_s": round(latency, 2), "speed": speed,
            "fast_segments_s": [[round(a1, 2), round(b1, 2)], [round(a2, 2), round(b2, 2)]],
            "real_time_s": round(normal, 2), "expected_total_s": round(normal + fast / speed, 2),
            "alert": log["alert"], "llm_reproduced_stored_output": log["llm_reproduced"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert", default="A07")
    parser.add_argument("--out", type=Path, default=ROOT / "media" / "W03_demo.mp4")
    parser.add_argument("--cap", type=int, default=600, help="hard cap on the raw recording, seconds")
    args = parser.parse_args()
    if args.out.name in PROTECTED:
        raise SystemExit(f"refusing to overwrite {args.out.name}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = args.out.with_name(args.out.stem + "_raw.mkv")
    events = ROOT / "media" / "w03_demo_events.json"
    events.unlink(missing_ok=True)
    t_start, t_stop = record(raw, args.alert, args.cap)
    if not events.exists():
        raise SystemExit("the demo program did not finish (no events file); raw recording kept for inspection")
    edit = render(raw, args.out, events, t_start, t_stop)
    edit["final_duration_s"] = round(duration(args.out), 2)
    if not MIN_TOTAL_S <= edit["final_duration_s"] <= 60.0:  # type: ignore[operator]
        raise SystemExit(f"final duration {edit['final_duration_s']} s is outside 30-60 s")
    args.out.with_suffix(".edit.json").write_text(json.dumps(edit, indent=1), encoding="utf-8")
    raw.unlink()
    print(json.dumps(edit, indent=1))


if __name__ == "__main__":
    main()
