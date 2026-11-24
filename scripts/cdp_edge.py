"""Minimal Chrome DevTools Protocol driver for a headless Edge/Chrome (no extra dependency).

Used by ``capture_w04_dashboard.py`` and ``record_w04_demo.py`` to look at the
*running* local services (Streamlit, Airflow, MLflow) exactly as a browser
renders them: navigate, click, type, wait for text, and take screenshots.
Nothing is drawn or edited; a screenshot is the browser's own rendering.
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

CANDIDATES = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
              Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
              Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")]


def find_browser() -> str:
    for name in ("msedge", "chromium", "chromium-browser", "google-chrome", "chrome"):
        exe = shutil.which(name)
        if exe:
            return exe
    for p in CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError("no Edge/Chrome/Chromium found")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Browser:
    def __init__(self, width: int = 1280, height: int = 720, scale: float = 1.0) -> None:
        self.port = free_port()
        self.profile = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.proc = subprocess.Popen(
            [find_browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
             "--hide-scrollbars", f"--remote-debugging-port={self.port}", f"--user-data-dir={self.profile.name}",
             f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ws_url = self._page_ws()
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024, open_timeout=30)
        self._id = 0
        self.width, self.height = width, height
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("Emulation.setDeviceMetricsOverride",
                  {"width": width, "height": height, "deviceScaleFactor": scale, "mobile": False})

    def _page_ws(self) -> str:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=2) as r:
                    for t in json.load(r):
                        if t.get("type") == "page":
                            return str(t["webSocketDebuggerUrl"])
            except OSError:
                pass
            time.sleep(0.3)
        raise TimeoutError("browser did not expose a page target")

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        my_id = self._id
        self.ws.send(json.dumps({"id": my_id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=120))
            if msg.get("id") == my_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return dict(msg.get("result", {}))

    def navigate(self, url: str) -> None:
        self.send("Page.navigate", {"url": url})

    def eval(self, js: str) -> Any:
        r = self.send("Runtime.evaluate", {"expression": js, "returnByValue": True, "awaitPromise": True})
        return r.get("result", {}).get("value")

    def wait_for(self, js: str, timeout: float = 60, poll: float = 0.25) -> Any:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                v = self.eval(js)
            except RuntimeError:
                v = None
            if v:
                return v
            time.sleep(poll)
        raise TimeoutError(f"condition not met within {timeout} s: {js[:120]}")

    def wait_text(self, text: str, timeout: float = 60) -> None:
        self.wait_for(f"document.body && document.body.innerText.includes({json.dumps(text)})", timeout)

    def click_xy(self, x: float, y: float) -> None:
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            self.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "clickCount": 1})

    def click_text(self, selector: str, text: str, timeout: float = 30) -> None:
        """Click the centre of the first ``selector`` element whose text contains ``text``."""
        js = (f"(() => {{ const els = [...document.querySelectorAll({json.dumps(selector)})]"
              f".filter(e => e.innerText && e.innerText.includes({json.dumps(text)}) && e.offsetParent !== null);"
              " if (!els.length) return null; const r = els[0].getBoundingClientRect();"
              " return [r.x + r.width / 2, r.y + r.height / 2]; })()")
        x, y = self.wait_for(js, timeout)
        self.click_xy(x, y)

    def type_text(self, text: str) -> None:
        self.send("Input.insertText", {"text": text})

    def press_enter(self) -> None:
        for kind in ("keyDown", "keyUp"):
            self.send("Input.dispatchKeyEvent", {"type": kind, "key": "Enter", "code": "Enter",
                                                 "windowsVirtualKeyCode": 13, "text": "\r" if kind == "keyDown" else ""})

    def screenshot(self, clip: dict[str, float] | None = None, fmt: str = "png") -> bytes:
        params: dict[str, Any] = {"format": fmt, "captureBeyondViewport": clip is not None}
        if fmt == "jpeg":
            params["quality"] = 88
        if clip:
            params["clip"] = {**clip, "scale": 1}
        return base64.b64decode(self.send("Page.captureScreenshot", params)["data"])

    def close(self) -> None:
        try:
            self.ws.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.profile.cleanup()
