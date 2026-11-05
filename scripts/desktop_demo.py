"""Scripted *real* desktop demo for Week 2 (Windows only).

Sends genuine OS mouse/keyboard events (user32 SendInput) to PyCharm and Edge
while FFmpeg (ddagrab, Desktop Duplication) records the screen. Nothing is
simulated: files are opened with PyCharm's own shortcuts, the dashboard is
started with PyCharm's Run command, and the browser is reached with Alt+Tab.
Before every key/mouse batch the foreground window is checked; if it is not
the expected application, the script aborts instead of typing elsewhere.

    python scripts/desktop_demo.py stage     # launch PyCharm + Edge, arrange windows (not recorded)
    python scripts/desktop_demo.py shot OUT  # one physical-resolution screenshot (for layout checks)
    python scripts/desktop_demo.py record    # perform the demo while recording media/W02_desktop_raw.mkv
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYCHARM = Path(r"C:\Program Files\JetBrains\PyCharm 2026.1\bin\pycharm64.exe")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ACTIONS_URL = "https://github.com/Kuntong705/ingen-robot-telemetry/actions?query=branch%3Aweek2"
DASH_TITLE = "W02 live dashboard"
EDGE_TITLE = "All workflows"  # GitHub's page title for the Actions list

user32 = ctypes.windll.user32
ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixels for coordinates and window rects

# ----------------------------------------------------------------------------- windows
EnumProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def find_windows(substring: str) -> list[int]:
    found: list[int] = []

    def cb(hwnd: int, _: int) -> bool:
        if user32.IsWindowVisible(hwnd) and substring.lower() in window_title(hwnd).lower():
            found.append(hwnd)
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return found


def wait_window(substring: str, timeout: float) -> int:
    end = time.time() + timeout
    while time.time() < end:
        hits = find_windows(substring)
        if hits:
            return hits[0]
        time.sleep(0.2)
    raise SystemExit(f"window '{substring}' not found")


def pycharm_window(timeout: float) -> int:
    """The PyCharm project window (Edge tabs on this repo also carry 'ingen-robot-telemetry' in the title)."""
    end = time.time() + timeout
    while time.time() < end:
        hits = [h for h in find_windows("ingen-robot-telemetry") if window_exe(h) == "pycharm64.exe"]
        if hits:
            return hits[0]
        time.sleep(0.2)
    raise SystemExit("PyCharm project window not found")


def activate(hwnd: int, maximize: bool = False) -> None:
    # ShowWindowAsync: the synchronous call can block for seconds while a background Edge window repaints
    if not (maximize and user32.IsZoomed(hwnd)):
        user32.ShowWindowAsync(hwnd, 3 if maximize else 9)
    # allow foreground change from a background process
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.4)


def foreground_title() -> str:
    return window_title(user32.GetForegroundWindow())


_WATCHDOG = None  # set while recording


def require_foreground(substring: str) -> None:
    if _WATCHDOG is not None and _WATCHDOG.violation.is_set():
        raise SystemExit(f"recording aborted and deleted: {_WATCHDOG.reason}")
    title = foreground_title()
    if substring.lower() not in title.lower():
        raise SystemExit(f"safety stop: expected '{substring}' in foreground, found '{title}'")


# ----------------------------------------------------------------------------- input (SendInput)
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


class _U(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _U)]


KEYUP, UNICODE = 0x0002, 0x0004
VK = {"ctrl": 0x11, "shift": 0x10, "alt": 0x12, "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "f10": 0x79,
      "pgdn": 0x22, "pgup": 0x21, "down": 0x28, "home": 0x24, "left": 0x25, "f4": 0x73}


def _send(*inputs: INPUT) -> None:
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def key(vk: int, up: bool = False) -> INPUT:
    # type 1 = INPUT_KEYBOARD; Java (PyCharm) also wants the hardware scan code
    return INPUT(1, _U(ki=KEYBDINPUT(vk, user32.MapVirtualKeyW(vk, 0), KEYUP if up else 0, 0, None)))


def hotkey(*names: str) -> None:
    codes = [VK[n] if n in VK else ord(n.upper()) for n in names]  # letters/digits: VK code == ASCII
    _send(*[key(c) for c in codes])
    time.sleep(_rng.uniform(0.06, 0.12))
    _send(*[key(c, up=True) for c in reversed(codes)])
    time.sleep(0.15)


def _alt_1() -> None:
    """Alt+1 toggles PyCharm's Project tool window (the tree shows the absolute project path)."""
    _send(key(VK["alt"]), key(0x31))
    time.sleep(0.05)
    _send(key(0x31, up=True), key(VK["alt"], up=True))
    time.sleep(0.3)


_rng = random.Random()


def pause(lo: float, hi: float) -> None:
    """A short, slightly irregular pause between actions."""
    time.sleep(_rng.uniform(lo, hi))


def type_text(text: str, mean_delay: float = 0.12) -> None:
    """Type at an ordinary pace: log-normal gaps between keys, a little longer after '_' and '.'."""
    for ch in text:
        _send(INPUT(1, _U(ki=KEYBDINPUT(0, ord(ch), UNICODE, 0, None))))
        time.sleep(_rng.uniform(0.03, 0.07))  # key held down
        _send(INPUT(1, _U(ki=KEYBDINPUT(0, ord(ch), UNICODE | KEYUP, 0, None))))
        gap = min(0.45, max(0.05, _rng.lognormvariate(math.log(mean_delay), 0.35)))
        if ch in "_.":
            gap += _rng.uniform(0.05, 0.15)
        time.sleep(gap)


def mouse_move(x: int, y: int, steps: int = 30, duration: float = 0.6) -> None:
    """Straight move with ease-in/ease-out (accelerate, then slow down near the target)."""
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    for i in range(1, steps + 1):
        f = i / steps
        f = f * f * (3 - 2 * f)
        user32.SetCursorPos(int(pt.x + (x - pt.x) * f), int(pt.y + (y - pt.y) * f))
        time.sleep(duration / steps)


def click(x: int, y: int) -> None:
    mouse_move(x, y)
    _send(INPUT(0, _U(mi=MOUSEINPUT(0, 0, 0, 0x0002, 0, None))), INPUT(0, _U(mi=MOUSEINPUT(0, 0, 0, 0x0004, 0, None))))
    time.sleep(0.2)


def scroll(clicks: int) -> None:
    for _ in range(abs(clicks)):
        _send(INPUT(0, _U(mi=MOUSEINPUT(0, 0, ctypes.c_ulong(-120 if clicks < 0 else 120).value, 0x0800, 0, None))))
        time.sleep(_rng.uniform(0.09, 0.2))


# ----------------------------------------------------------------------------- ffmpeg
def ffmpeg_exe() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_demo_video import find_ffmpeg

    exe = Path(find_ffmpeg())
    gyan = sorted((Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
    return str(gyan[0]) if gyan else str(exe)  # the Gyan full build has ddagrab


def screenshot(out: Path) -> None:
    subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "ddagrab=framerate=5,hwdownload,format=bgra", "-frames:v", "1", str(out)], check=True)


# ----------------------------------------------------------------------------- steps
def stage() -> None:
    if not find_windows("ingen-robot-telemetry"):
        subprocess.Popen([str(PYCHARM), str(ROOT)])
    pc = pycharm_window(180)
    if not find_windows(EDGE_TITLE):
        subprocess.Popen([str(EDGE), "--new-window", ACTIONS_URL])
    edge = wait_window(EDGE_TITLE, 60)
    activate(edge, maximize=True)
    activate(pc, maximize=True)
    print("PyCharm:", window_title(pc))
    print("Edge:", window_title(edge))


def prep() -> None:
    """Not recorded: Edge back to the Actions list, PyCharm editor tabs closed (left by a previous take)."""
    edges = find_windows(EDGE_TITLE) or find_windows("Run NASA C-MAPSS")
    if not edges:
        raise SystemExit("Edge Actions window not found")
    activate(edges[0], maximize=True)
    if EDGE_TITLE.lower() not in foreground_title().lower():
        require_foreground("Run NASA C-MAPSS")
        hotkey("alt", "left")  # browser Back
        end = time.time() + 10
        while EDGE_TITLE.lower() not in foreground_title().lower() and time.time() < end:
            time.sleep(0.3)
    require_foreground(EDGE_TITLE)
    pc = pycharm_window(5)
    activate(pc, maximize=True)
    require_foreground("ingen-robot-telemetry")
    click(1100, 400)  # editor area
    for _ in range(4):
        if "w02_live_dashboard.py" not in foreground_title() and "early_warning.py" not in foreground_title():
            break
        hotkey("ctrl", "f4")  # close the active editor tab
        time.sleep(0.4)
    print("prepared:", foreground_title())


def log_event(events: list[tuple[float, str]], t0: float, text: str) -> None:
    events.append((round(time.perf_counter() - t0, 2), text))
    print(f"[{events[-1][0]:5.1f}s] {text}", flush=True)


def foreground_exe() -> str:
    return window_exe(user32.GetForegroundWindow())


def window_exe(hwnd: int) -> str:
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(512)
        size = wt.DWORD(512)
        ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        return Path(buf.value).name.lower()
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def claude_windows_open() -> list[str]:
    """Visible, non-minimised top-level windows of the Claude desktop app."""
    open_: list[str] = []

    def cb(hwnd: int, _: int) -> bool:
        if (user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and window_title(hwnd)
                and window_exe(hwnd) == "claude.exe"):
            open_.append(window_title(hwnd))
        return True

    user32.EnumWindows(EnumProc(cb), 0)
    return open_


def wait_claude_minimised(timeout: float = 120.0, settle: float = 5.0) -> None:
    """Gate before any desktop action: the user minimises the Claude app after sending 'start'."""
    end = time.time() + timeout
    while claude_windows_open():
        if time.time() > end:
            raise SystemExit("Claude window still open; nothing was touched")
        time.sleep(0.5)
    print(f"Claude minimised; waiting {settle:.0f}s for hands off the mouse/keyboard", flush=True)
    time.sleep(settle)


def foreground_allowed() -> bool:
    """Only PyCharm, the dashboard and this repository's GitHub pages may be in front while recording."""
    if not user32.GetForegroundWindow():
        return True  # transient state during a window switch
    exe, title = foreground_exe(), foreground_title().lower()
    if exe == "pycharm64.exe":
        return True
    if exe.startswith("python") and exe.endswith(".exe"):  # python.exe / pythonw.exe / python3.11.exe
        return DASH_TITLE.lower() in title or title in ("", "matplotlib", "tk")  # Tk frames while closing
    if exe == "msedge.exe":
        return EDGE_TITLE.lower() in title or "ingen-robot-telemetry" in title
    return exe in ("explorer.exe",) and title == ""  # bare desktop/task-switcher frames carry no content


class Watchdog(threading.Thread):
    """Stops the recording and deletes the raw file the moment any other window comes to the front."""

    def __init__(self, rec: subprocess.Popen, raw: Path) -> None:
        super().__init__(daemon=True)
        self.rec, self.raw = rec, raw
        self.violation = threading.Event()
        self.stopped = threading.Event()
        self.reason = ""

    def run(self) -> None:
        while not self.stopped.is_set():
            if not foreground_allowed():
                self.reason = f"{foreground_exe()} in foreground (title {foreground_title()[:40]!r})"
                self.violation.set()
                self.rec.kill()
                self.rec.wait(timeout=10)
                self.raw.unlink(missing_ok=True)
                return
            time.sleep(0.1)


def record(raw: Path, run_title_xy: tuple[int, int]) -> list[tuple[float, str]]:
    pc = pycharm_window(5)
    edge = wait_window(EDGE_TITLE, 5)
    # Z-order before recording: Edge (Actions) directly behind PyCharm, so one Alt+Tab reaches it
    activate(edge, maximize=True)
    activate(pc, maximize=True)
    require_foreground("ingen-robot-telemetry")
    user32.SetCursorPos(1100, 560)
    rec = subprocess.Popen([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "ddagrab=framerate=20:draw_mouse=1,hwdownload,format=bgra", "-t", "110",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(raw)],
                           stdin=subprocess.PIPE, text=True)
    watchdog = Watchdog(rec, raw)
    watchdog.start()
    global _WATCHDOG
    _WATCHDOG = watchdog
    t0 = time.perf_counter()
    events: list[tuple[float, str]] = []
    try:
        time.sleep(1.0)
        require_foreground("ingen-robot-telemetry")
        click(1300, 850)  # empty editor area: gives PyCharm's Java window keyboard focus
        pause(0.5, 0.8)
        require_foreground("ingen-robot-telemetry")
        log_event(events, t0, "Ctrl+Shift+N: open models/early_warning.py")
        hotkey("ctrl", "shift", "n")
        pause(0.6, 0.9)
        type_text("early_warning.py")
        pause(0.5, 0.8)
        hotkey("enter")
        pause(1.0, 1.3)
        require_foreground("early_warning.py")
        mouse_move(1100, 560, duration=0.5)
        pause(0.2, 0.4)
        scroll(-8)
        pause(0.9, 1.2)
        log_event(events, t0, "Ctrl+Shift+N: open scripts/w02_live_dashboard.py")
        hotkey("ctrl", "shift", "n")
        pause(0.6, 0.9)
        type_text("w02_live_dashboard.py")
        pause(0.5, 0.8)
        hotkey("enter")
        pause(1.0, 1.3)
        require_foreground("w02_live_dashboard.py")
        log_event(events, t0, "Ctrl+Shift+F10: run the dashboard from PyCharm")
        hotkey("ctrl", "shift", "f10")
        end = time.time() + 90
        while not find_windows(DASH_TITLE) and time.time() < end:
            time.sleep(0.3)
        dash = wait_window(DASH_TITLE, 1)
        log_event(events, t0, "dashboard window open (live computation)")
        while user32.IsWindow(dash) and time.time() < end + 90:
            time.sleep(0.2)
        log_event(events, t0, "dashboard finished and closed itself")
        pause(1.0, 1.3)  # PyCharm shows "process finished with exit code 0"
        require_foreground("ingen-robot-telemetry")
        # Direct window switch: Alt+Tab would show the task switcher with thumbnails of every window.
        log_event(events, t0, "switch to Edge (GitHub Actions)")
        activate(edge, maximize=True)
        log_event(events, t0, "Edge in front")
        pause(0.3, 0.5)
        require_foreground(EDGE_TITLE)
        mouse_move(*run_title_xy, duration=0.7)
        log_event(events, t0, "pointer on run #11")
        pause(0.3, 0.5)
        log_event(events, t0, "click: open CI run #11")
        click(*run_title_xy)
        time.sleep(8.0)  # the run page loads; the renderer keeps >= 4 s of the loaded Success page
        log_event(events, t0, "end")
    finally:
        watchdog.stopped.set()
        if rec.poll() is None and rec.stdin:
            rec.stdin.write("q")
            rec.stdin.flush()
        rec.wait(timeout=30)
        if watchdog.violation.is_set():
            raw.unlink(missing_ok=True)
            raise SystemExit(f"recording aborted and deleted: {watchdog.reason}")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["wait-minimised", "stage", "prep", "shot", "hide-project-tree", "record"])
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--run-xy", type=int, nargs=2, default=(700, 488))
    args = parser.parse_args()
    if args.cmd == "wait-minimised":
        wait_claude_minimised()
    elif args.cmd == "stage":
        stage()
    elif args.cmd == "prep":
        prep()
    elif args.cmd == "hide-project-tree":
        activate(pycharm_window(5), maximize=True)
        require_foreground("ingen-robot-telemetry")
        _alt_1()
    elif args.cmd == "record":
        out = args.out or ROOT / "media" / "W02_desktop_raw.mkv"
        events = record(out, tuple(args.run_xy))
        import json

        out.with_suffix(".events.json").write_text(json.dumps(events, indent=1), encoding="utf-8")
    elif args.cmd == "shot":
        screenshot(args.out)


if __name__ == "__main__":
    main()
