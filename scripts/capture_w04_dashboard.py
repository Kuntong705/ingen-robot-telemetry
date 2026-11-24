"""Screenshot one alert of the running Week 4 dashboard for the capstone deck (slide 7).

Starts the Streamlit dashboard on the full study data (``configs/pipeline.yaml``,
synthetic fleets), opens it in a headless Edge/Chrome, switches to the alert
queue, expands the newest alert and saves the browser's own rendering of that
panel to ``docs/figures/w04/dashboard_alert.png``. Nothing is edited.

    python scripts/capture_w04_dashboard.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cdp_edge import Browser, free_port

OUT = ROOT / "docs" / "figures" / "w04" / "dashboard_alert.png"


def main() -> None:
    port = free_port()
    env = {**os.environ, "FLEET_PIPELINE_CONFIG": str(ROOT / "configs" / "pipeline.yaml")}
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                            "--server.headless", "true", "--server.port", str(port),
                            "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    browser = None
    try:
        deadline = time.time() + 90
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)
        # a tall window (no scrolling, so the sticky header never covers the panel); the sidebar is collapsed
        # so the alert panel gets the full width
        browser = Browser(width=1000, height=1600, scale=2)
        browser.navigate(f"http://127.0.0.1:{port}/")
        browser.wait_text("Robots to pull", timeout=120)
        browser.eval("(() => { const b = document.querySelector('[data-testid=\"stSidebarCollapseButton\"] button');"
                     " if (b) b.click(); })()")
        time.sleep(1)
        browser.click_text('[role="tab"]', "Alert queue")
        browser.wait_text("Each explanation is written by a fixed template")
        browser.click_text('[data-testid="stExpander"] summary', "UTC |")
        browser.wait_text("Automatic evidence check:")
        # the driver table is drawn on a canvas: give it time to paint
        browser.wait_for("(() => { const c = document.querySelector('[data-testid=\"stExpander\"] canvas');"
                         " return c && c.width > 0; })()")
        time.sleep(4)
        rect = browser.eval(
            "(() => { const e = document.querySelector('[data-testid=\"stExpander\"]');"
            " const r = e.getBoundingClientRect(); return {x: r.x, y: r.y + window.scrollY,"
            " width: r.width, height: r.height}; })()")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(browser.screenshot(clip={"x": rect["x"] - 8, "y": rect["y"] - 8,
                                                 "width": rect["width"] + 16, "height": rect["height"] + 16}))
        print(f"wrote {OUT.relative_to(ROOT)}")
    finally:
        if browser:
            browser.close()
        app.terminate()


if __name__ == "__main__":
    main()
