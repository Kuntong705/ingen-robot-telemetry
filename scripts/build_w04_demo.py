"""Week 4 demo video from real screenshots of the running stack (SYNTHETIC data).

Two steps, so nothing ever waits on the browser for an unbounded time:

``capture``  Takes screenshots of the live Docker Compose stack (Airflow, Streamlit,
             MLflow) in a headless Edge/Chrome with an English UI and UTC clocks.
             Every wait is bounded (each screenshot at most 15 s, three tries). It
             triggers one real DAG run and screenshots it every ~2.5 s until it
             finishes, and screenshots the earlier missing-partition run. Writes the
             frames and ``manifest.json`` (URL, capture time) to ``--frames``.
``assemble`` Builds ``media/W04_demo.mp4`` (1920x1080) from those frames only. Each
             scene is labelled: "STATIC SCREENSHOT", or "REAL RUN - Nx SPEED (real
             T s)" for the DAG-run sequence, or "TITLE CARD". Captions and labels are
             drawn onto the frames (no overlap); scenes cross-fade for 0.4 s.

    python scripts/build_w04_demo.py capture --project fleetfresh --compose-dir <clone> --run-date 2026-11-19
    python scripts/build_w04_demo.py assemble
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cdp_edge import Browser
from render_demo_video import find_ffmpeg

AIRFLOW, DASH, MLFLOW = "http://localhost:8080", "http://localhost:8501", "http://localhost:5000"
DAG = "fleet_health_daily"
W, H = 1920, 1080
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT = "C:/Windows/Fonts/segoeui.ttf"
FRAMES = ROOT / "media" / "w04_demo_frames"
BANNER = "SYNTHETIC DATA - simulated fleets, no real InGen robot"


# --------------------------------------------------------------------------- capture
class Stack:
    def __init__(self, project: str, compose_dir: Path) -> None:
        self.project, self.dir, self.token = project, compose_dir, ""

    def login(self) -> None:
        """API token for the local, generated admin account (never printed or stored)."""
        out = subprocess.run(["docker", "compose", "-p", self.project, "exec", "-T", "airflow", "cat",
                              "/opt/airflow/simple_auth_manager_passwords.json.generated"], cwd=self.dir,
                             capture_output=True, text=True, check=True, timeout=60).stdout
        user, password = next(iter(json.loads(out).items()))
        req = urllib.request.Request(f"{AIRFLOW}/auth/token", method="POST",
                                     data=json.dumps({"username": user, "password": password}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.token = json.load(r)["access_token"]

    def api(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(f"{AIRFLOW}/api/v2{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return dict(json.load(r))


class Camera:
    def __init__(self, browser: Browser, out: Path) -> None:
        self.b, self.out = browser, out
        self.shots: list[dict[str, Any]] = []

    def shot(self, name: str, note: str = "") -> dict[str, Any]:
        """One screenshot, at most 15 s per try and three tries; raises instead of waiting forever."""
        for attempt in range(3):
            try:
                data = self.b.screenshot(fmt="png", timeout=15)
                break
            except TimeoutError:
                if attempt == 2:
                    raise
                self.b.post("Runtime.evaluate", {"expression": "window.scrollBy(0, 1); window.scrollBy(0, -1);"})
                time.sleep(2)
        path = self.out / f"{name}.png"
        path.write_bytes(data)
        url = self.b.eval("location.href")
        rec = {"file": path.name, "url": url, "captured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
               "t": time.time(), "note": note}
        self.shots.append(rec)
        print(f"  {name}  {url}", flush=True)
        return rec


def open_tab(browser: Browser, tab: str, expect: str) -> None:
    """Click a Streamlit tab until its content shows: at most 3 clicks, 20 s each."""
    for attempt in range(3):
        browser.click_text('[role="tab"]', tab)
        try:
            browser.wait_text(expect, 20)
            return
        except TimeoutError:
            if attempt == 2:
                raise


def capture_dashboard(out: Path, cam: Camera) -> None:
    """Pull list, unit drill-down and alert queue, each painted before its screenshot (bounded waits)."""
    browser = Browser(1280, 720, scale=1.5)
    dash = Camera(browser, out)
    try:
        browser.navigate(DASH)
        browser.wait_text("Robots to pull", 120)
        time.sleep(4)
        dash.shot("dash_pull")
        open_tab(browser, "Unit drill-down", "Robot (highest current risk first)")
        time.sleep(4)
        # Plotly and the canvas grid lay themselves out on resize: send one, then let them paint
        browser.eval("window.dispatchEvent(new Event('resize'))")
        time.sleep(5)
        dash.shot("dash_unit")
        open_tab(browser, "Alert queue", "Each explanation is written by a fixed template")
        browser.click_text('[data-testid="stExpander"] summary', "UTC |")
        browser.wait_text("Automatic evidence check:", 30)
        time.sleep(3)
        browser.eval("window.dispatchEvent(new Event('resize'))")
        time.sleep(5)
        dash.shot("dash_alert")
    finally:
        browser.close()
    names = {s["file"] for s in dash.shots}
    cam.shots = [s for s in cam.shots if s["file"] not in names] + dash.shots


def recapture_dashboard(args: argparse.Namespace) -> None:
    """Replace only the three dashboard screenshots of an existing capture (no new DAG run)."""
    path = args.frames / "manifest.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    cam = Camera.__new__(Camera)
    cam.shots = meta["shots"]
    capture_dashboard(args.frames, cam)
    meta["shots"] = cam.shots
    meta["dashboard_recaptured_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("replaced dash_pull, dash_unit, dash_alert")


def capture(args: argparse.Namespace) -> None:
    out: Path = args.frames
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):   # frames of an earlier capture: never mix two captures
        old.unlink()
    stack = Stack(args.project, args.compose_dir)
    stack.login()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=args.compose_dir, capture_output=True,
                            text=True, check=True).stdout.strip()
    runs = stack.api("GET", f"/dags/{DAG}/dagRuns?limit=100&order_by=-run_after")["dag_runs"]
    failed = next((r for r in runs if r["state"] == "failed"), None)
    if failed is None:
        raise SystemExit("no failed (missing-partition) run found; create it first (see the demo script)")
    browser = Browser(1280, 720, scale=1.5)   # 1920x1080 pixels
    cam = Camera(browser, out)
    meta: dict[str, Any] = {"commit": commit, "captured_from": "local Docker Compose stack (synthetic quickstart fleet)"}
    try:
        browser.send("Network.enable")
        browser.send("Network.setCookie", {"name": "_token", "value": stack.token, "url": AIRFLOW})
        browser.send("Network.disable")
        # 1) a real DAG run, triggered now and followed until it finishes
        stack.api("PATCH", f"/dags/{DAG}", {"is_paused": False})
        logical = (datetime.fromisoformat(args.run_date).replace(tzinfo=UTC)
                   + timedelta(days=1, hours=2, minutes=args.logical_minute))
        run_id = f"w04_demo__{datetime.now(UTC):%Y%m%dT%H%M%S}"
        stack.api("POST", f"/dags/{DAG}/dagRuns", {"dag_run_id": run_id, "logical_date": logical.isoformat()})
        browser.navigate(f"{AIRFLOW}/dags/{DAG}/runs/{run_id}")
        browser.wait_text(run_id, 60)
        time.sleep(1)
        start, state, k = time.time(), "queued", 0
        while True:
            # the run page does not refresh itself: reload it before every screenshot (bounded waits)
            browser.navigate(f"{AIRFLOW}/dags/{DAG}/runs/{run_id}")
            browser.wait_text(run_id, 20)
            time.sleep(1.2)
            cam.shot(f"run_{k:03d}", note=f"run state {state}")
            k += 1
            state = stack.api("GET", f"/dags/{DAG}/dagRuns/{run_id}")["state"]
            if state in ("success", "failed"):
                break
            if time.time() - start > 600:
                raise TimeoutError("DAG run did not finish within 10 min")
            time.sleep(1.0)
        time.sleep(3)
        browser.navigate(f"{AIRFLOW}/dags/{DAG}/runs/{run_id}")
        browser.wait_text(run_id, 60)
        time.sleep(3)
        cam.shot("run_final", note=f"run state {state}")
        meta["triggered_run"] = {"dag_run_id": run_id, "logical_date": logical.isoformat(), "final_state": state,
                                 "processed_date": args.run_date}
        # 2) dashboard, in a fresh browser (after many Airflow pages, Plotly and the canvas grid did not paint)
        capture_dashboard(out, cam)
        # 3) the earlier missing-partition run and its sensor log
        browser.navigate(f"{AIRFLOW}/dags/{DAG}/runs/{failed['dag_run_id']}")
        browser.wait_text(failed["dag_run_id"], 60)
        time.sleep(3)
        cam.shot("missing_run")
        browser.navigate(f"{AIRFLOW}/dags/{DAG}/runs/{failed['dag_run_id']}/tasks/fari.wait_landing")
        try:
            browser.wait_text("AirflowSensorTimeout", 45)
            time.sleep(2)
            cam.shot("missing_log")
        except TimeoutError:
            print("  sensor log text not shown within 45 s; scene uses the run page only", flush=True)
        meta["missing_partition_run"] = {"dag_run_id": failed["dag_run_id"], "state": failed["state"],
                                         "run_after": failed.get("run_after")}
        # 4) MLflow registry
        browser.navigate(f"{MLFLOW}/#/models/fleet_early_warning")
        browser.wait_text("fleet_early_warning", 60)
        time.sleep(3)
        # close MLflow's assistant side panel if it is open (one try; the page is shown either way)
        browser.eval("(() => { const b = [...document.querySelectorAll('button')].find(x => "
                     "(x.getAttribute('aria-label') || '').toLowerCase().includes('close')); if (b) b.click(); })()")
        time.sleep(2)
        cam.shot("mlflow")
    finally:
        browser.close()
    meta["shots"] = cam.shots
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(cam.shots)} screenshots and manifest.json to {out}")


# --------------------------------------------------------------------------- assemble
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def box_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, f: ImageFont.FreeTypeFont,
             fill: tuple[int, int, int, int], anchor: str) -> tuple[int, int, int, int]:
    """Draw text on a dark box; ``anchor`` is 'top', 'bottom-left' or 'bottom-right'. Returns the box."""
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=f)
    tw, th = x1 - x0, y1 - y0
    pad_x, pad_y = 18, 12
    if anchor == "top":
        x, y = (W - tw) // 2, xy[1]
    elif anchor == "bottom-left":
        x, y = xy[0], H - xy[1] - th
    else:
        x, y = W - xy[0] - tw, H - xy[1] - th
    rect = (x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y)
    draw.rectangle(rect, fill=fill)
    draw.text((x - x0, y - y0), text, font=f, fill="white")
    return rect


def decorate(img: Image.Image, caption: str, badge: str) -> Image.Image:
    img = img.convert("RGB").resize((W, H), Image.LANCZOS) if img.size != (W, H) else img.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    cap = box_text(d, (0, 22), caption, font(38), (15, 23, 42, 215), "top")
    left = box_text(d, (24, 26), BANNER, font(26), (127, 29, 29, 225), "bottom-left")
    right = box_text(d, (24, 26), badge, font(26), (30, 64, 175, 225), "bottom-right")
    if left[2] >= right[0] or cap[0] < 0 or cap[2] > W:
        raise ValueError(f"captions overlap or overflow: {caption!r} / {badge!r}")
    return img


def card(lines: list[tuple[str, int, bool]]) -> Image.Image:
    img = Image.new("RGB", (W, H), (15, 23, 42))
    d = ImageDraw.Draw(img)
    heights = [d.textbbox((0, 0), t, font=font(s, b))[3] + 28 for t, s, b in lines]
    y = (H - sum(heights)) // 2
    for (t, s, b), h in zip(lines, heights, strict=True):
        f = font(s, b)
        x0, _, x1, _ = d.textbbox((0, 0), t, font=f)
        d.text(((W - (x1 - x0)) // 2 - x0, y), t, font=f, fill="white")
        y += h
    return img


def assemble(args: argparse.Namespace) -> None:
    src: Path = args.frames
    meta = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    shots = {s["file"].removesuffix(".png"): s for s in meta["shots"]}

    def load(n: str) -> Image.Image:
        return Image.open(src / f"{n}.png")

    run = sorted(n for n in shots if n.startswith("run_") and n != "run_final")
    real_run = shots[run[-1]]["t"] - shots[run[0]]["t"]
    run_budget = 8.0
    speed = max(1, math.ceil(real_run / run_budget))
    state = meta["triggered_run"]["final_state"]
    commit = meta["commit"]
    static = "STATIC SCREENSHOT"
    # scene = (caption, badge, [(image, seconds)])
    scenes: list[tuple[str, str, list[tuple[Image.Image, float]]]] = [
        ("Week 4: daily scoring pipeline, dashboard and model registry", "TITLE CARD",
         [(card([("ingen_fleet_health v1.0 - Week 4 demo", 64, True),
                 ("Airflow DAG  |  Fleet health dashboard  |  Alert explanations  |  MLflow registry", 34, False),
                 (f"SYNTHETIC DATA - local Docker Compose stack, commit {commit}", 32, False),
                 ("Screenshots of real runs; no real InGen robot", 32, False)]), 3.0)]),
        ("Airflow: a real daily DAG run, both platforms", f"REAL RUN - {speed}x SPEED (real {real_run:.0f} s)",
         [(load(n), (shots[run[i + 1]]["t"] - shots[n]["t"]) / speed if i + 1 < len(run) else 0.6)
          for i, n in enumerate(run)]),
        (f"Airflow: run finished - {state}, all tasks green", static, [(load("run_final"), 3.5)]),
        ("Dashboard: which robots to pull before the next shift, and why", static, [(load("dash_pull"), 8.0)]),
        ("Unit drill-down: risk against the alert level", static, [(load("dash_unit"), 7.0)]),
        ("Alert queue: template explanation + automatic evidence check", static, [(load("dash_alert"), 8.0)]),
    ]
    if "missing_log" in shots:
        scenes += [("Missing Fari partition: sensor waited, then failed loudly", static, [(load("missing_run"), 4.5)]),
                   ("Sensor log: AirflowSensorTimeout, nothing written for that date", static,
                    [(load("missing_log"), 4.5)])]
    else:
        scenes += [("Missing Fari partition: sensor timed out, run failed, nothing written", static,
                    [(load("missing_run"), 9.0)])]
    scenes += [
        ("MLflow registry: fleet_early_warning v1, pinned by the pipeline", static, [(load("mlflow"), 5.0)]),
        ("All data synthetic - not a real deployment", "TITLE CARD",
         [(card([("All robots, faults and numbers are simulated", 48, True),
                  ("Screenshots and the accelerated run are from one real run of the stack", 32, False),
                  (f"commit {commit}  |  recorded {meta['shots'][0]['captured_utc'][:10]}", 32, False)]), 2.5)]),
    ]
    fade, fps = 0.4, 25
    timeline: list[tuple[Image.Image, float]] = []
    cut = []
    t_out = 0.0
    prev_raw: Image.Image | None = None
    for si, (caption, badge, frames) in enumerate(scenes):
        start = t_out
        first_raw = frames[0][0].convert("RGB").resize((W, H), Image.LANCZOS)
        if prev_raw is not None:
            n = int(fade * fps)
            for j in range(1, n + 1):
                blend = Image.blend(prev_raw, first_raw, j / (n + 1))
                timeline.append((decorate(blend, caption, badge), 1 / fps))
                t_out += 1 / fps
        for img, secs in frames:
            timeline.append((decorate(img, caption, badge), secs))
            t_out += secs
        prev_raw = frames[-1][0].convert("RGB").resize((W, H), Image.LANCZOS)
        cut.append({"scene": si + 1, "caption": caption, "label": badge, "from_s": round(start, 2),
                    "to_s": round(t_out, 2), "frames": len(frames)})
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        lines = []
        for i, (img, secs) in enumerate(timeline):
            name = f"v{i:05d}.jpg"
            img.save(work / name, quality=93)
            lines += [f"file '{name}'", f"duration {secs:.4f}"]
        lines.append(f"file 'v{len(timeline) - 1:05d}.jpg'")
        (work / "list.txt").write_text("\n".join(lines), encoding="utf-8")
        out = args.out
        subprocess.run([find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", "list.txt", "-t", f"{t_out:.3f}", "-vf", f"fps={fps},format=yuv420p",
                        "-c:v", "libx264", "-crf", "18",
                        "-movflags", "+faststart", "-metadata", f"comment={BANNER}; commit {commit}", str(out)],
                       cwd=work, check=True, timeout=600)
    edit = {"video": out.name, "resolution": f"{W}x{H}", "fps": fps, "total_s": round(t_out, 2),
            "accelerated_segment": {"speed": speed, "real_s": round(real_run, 1), "frames": len(run)},
            "transitions": f"{fade} s cross-fade between scenes", "source_frames": str(src.relative_to(ROOT)),
            "triggered_run": meta["triggered_run"], "missing_partition_run": meta["missing_partition_run"],
            "scenes": cut}
    out.with_suffix(".edit.json").write_text(json.dumps(edit, indent=2), encoding="utf-8")
    print(json.dumps({k: edit[k] for k in ("video", "resolution", "total_s")}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--project", default="fleetfresh")
    c.add_argument("--compose-dir", type=Path, default=ROOT)
    c.add_argument("--run-date", required=True, help="processed date D (logical date D + 1 day 02:00 UTC)")
    c.add_argument("--logical-minute", type=int, default=0,
                   help="minutes after 02:00 (Airflow logical dates are unique; the processed date is the same)")
    c.add_argument("--frames", type=Path, default=FRAMES)
    r = sub.add_parser("recapture-dashboard")
    r.add_argument("--frames", type=Path, default=FRAMES)
    a = sub.add_parser("assemble")
    a.add_argument("--frames", type=Path, default=FRAMES)
    a.add_argument("--out", type=Path, default=ROOT / "media" / "W04_demo.mp4")
    args = p.parse_args()
    {"capture": capture, "recapture-dashboard": recapture_dashboard, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
