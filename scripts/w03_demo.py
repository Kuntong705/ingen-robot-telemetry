"""Live Week 3 demo: a fault alert and its explanation, end to end, on synthetic data.

What runs live in this program (nothing is retrained):

1. the alert: the unit's stored out-of-fold early-fusion scores (written by the
   Week 3 fusion run) are read and compared with the fold's alert threshold;
2. the evidence pack of that alert (tree SHAP attributions, values now and 72 h
   earlier, linked operator notes) from the Week 3 explain run;
3. the template summary and the local LLM summary (Qwen2.5-1.5B-Instruct, CPU,
   the same greedy settings as the explain run) are generated now; the LLM's
   tokens are streamed as they are produced;
4. the faithfulness checker (v2) scores both summaries now, the LLM text is
   compared with the stored Week 3 output, and the AI-assisted evidence-review
   verdict for this alert is shown.

Wall-clock times of the steps are written to ``--events`` so that
``scripts/record_w03_screen_demo.py`` can mark any accelerated segment.

    python scripts/w03_demo.py --alert A07 --events media/w03_demo_events.json
"""

from __future__ import annotations

import time

T0 = time.time()  # before the heavy imports, so the start-up wait can be marked in the video

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any


def prepare_console() -> None:
    """For the screen recording (Windows only): larger Consolas, then restore the window size.

    Driven by W03_DEMO_FONT (font height, px) and W03_DEMO_RECT ("x,y,w,h"), which
    scripts/record_w03_screen_demo.py sets. Changing the font resizes a console
    window, so the window is moved back to the recorded rectangle afterwards.
    """
    height = int(os.environ.get("W03_DEMO_FONT", "0"))
    if os.name != "nt" or height <= 0:
        return
    import ctypes
    from ctypes import wintypes

    class ConsoleFontInfoEx(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.ULONG), ("nFont", wintypes.DWORD), ("X", wintypes.SHORT),
                    ("Y", wintypes.SHORT), ("FontFamily", wintypes.UINT), ("FontWeight", wintypes.UINT),
                    ("FaceName", wintypes.WCHAR * 32)]

    info = ConsoleFontInfoEx()
    info.cbSize = ctypes.sizeof(ConsoleFontInfoEx)
    info.Y = height
    info.FontFamily = 54
    info.FontWeight = 400
    info.FaceName = "Consolas"
    kernel32 = ctypes.windll.kernel32
    kernel32.SetCurrentConsoleFontEx(kernel32.GetStdHandle(-11), False, ctypes.byref(info))
    rect = os.environ.get("W03_DEMO_RECT")
    if rect:
        x, y, w, h = (int(v) for v in rect.split(","))
        ctypes.windll.user32.MoveWindow(kernel32.GetConsoleWindow(), x, y, w, h, True)


prepare_console()

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights come from the local cache only
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import pandas as pd

from ingen_fleet_health.experiments.explain_run import evidence_from_dict
from ingen_fleet_health.explain.alert_summaries import (
    LocalLLMSummariser,
    TemplateSummariser,
    build_prompt,
    describe_attribution,
)
from ingen_fleet_health.explain.faithfulness import check_summary

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "w03"
OOF = ROOT / "data" / "w02" / "predictions" / "fusion_oof.parquet"
WIDTH = 108
EVENTS: list[tuple[str, float]] = [("python_start", T0)]


def event(name: str) -> None:
    EVENTS.append((name, time.time()))


def say(text: str = "", indent: int = 2, pause: float = 0.0) -> None:
    pad = " " * indent
    print(textwrap.fill(text, WIDTH, initial_indent=pad, subsequent_indent=pad) if text else "", flush=True)
    if pause:
        time.sleep(pause)


def head(text: str, pause: float = 0.4) -> None:
    print(flush=True)
    print(text, flush=True)
    time.sleep(pause)


def generate_streaming(llm: LocalLLMSummariser, ev: Any) -> str:
    """LocalLLMSummariser.summarise with the tokens printed as they are generated."""
    from transformers import TextStreamer

    class Wrapped(TextStreamer):
        col = 4

        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            for piece in re.split(r"(\s+)", text):
                if not piece:
                    continue
                if piece.isspace():
                    if self.col > 4:
                        sys.stdout.write(" ")
                        self.col += 1
                    continue
                if self.col + len(piece) > WIDTH + 2 and self.col > 4:
                    sys.stdout.write("\n    ")
                    self.col = 4
                sys.stdout.write(piece)
                self.col += len(piece)
            if stream_end:
                sys.stdout.write("\n")
            sys.stdout.flush()

    system, user = build_prompt(ev)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm.tokenizer(prompt, return_tensors="pt")
    streamer = Wrapped(llm.tokenizer, skip_prompt=True, skip_special_tokens=True)
    with llm.torch.no_grad():
        out = llm.model.generate(**inputs, max_new_tokens=llm.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.05, pad_token_id=llm.tokenizer.eos_token_id,
                                 streamer=streamer)
    text = str(llm.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert", default="A07")
    parser.add_argument("--events", type=Path, default=ROOT / "media" / "w03_demo_events.json")
    args = parser.parse_args()
    ex = json.loads((RESULTS / "alert_explanations.json").read_text(encoding="utf-8"))
    review = json.loads((RESULTS / "llm_faithfulness_review.json").read_text(encoding="utf-8"))
    a = next(x for x in ex["alerts"] if x["alert_id"] == args.alert)
    e = a["evidence"]
    ev = evidence_from_dict(e)
    gen = review["generator"]

    event("first_output")
    head("[1/4] Fault alert: the Week 3 early-fusion model's score for one unseen robot (stored fold scores, no retraining)")
    oof = pd.read_parquet(OOF, filters=[("model", "==", "early_fusion"), ("unit_id", "==", e["unit_id"])])
    t_alert = pd.Timestamp(e["t"], tz="UTC")
    win = oof[(oof["t"] > t_alert - pd.Timedelta(hours=6)) & (oof["t"] <= t_alert)].sort_values("t")
    for _, row in win.iterrows():
        flag = "  <- ALERT" if row["score"] >= row["thr"] else ""
        say(f"{row['t']:%Y-%m-%d %H:%M} UTC   risk {100 * row['score']:5.2f} %   threshold {100 * row['thr']:4.2f} %{flag}",
            pause=0.25)
    last = win.iloc[-1]
    assert last["score"] >= last["thr"] and abs(last["score"] - e["risk"]) < 1e-9, "stored score must raise this alert"
    say(f"ALERT {e['unit_id']} at {e['t']} UTC: chance of a qualifying fault onset within 72 h "
        f"{100 * e['risk']:.1f} % >= alert threshold {100 * e['threshold']:.1f} %.", pause=1.5)

    head("[2/4] Evidence pack (exact tree SHAP; values now and 72 h earlier)")
    for x in ev.positive[:3]:
        say(f"raises risk  {x.shap:+.2f}  {describe_attribution(x)}", pause=0.4)
    for x in ev.negative[:1]:
        say(f"lowers risk  {x.shap:+.2f}  {describe_attribution(x)}", pause=0.4)
    for n in ev.notes[:2]:
        say(f"operator note {n.age_hours:.1f} h before: \"{n.text}\" (classifier: {n.predicted_class.replace('_', ' ')})",
            pause=0.4)
    time.sleep(1.5)

    head("[3/4] Explanations generated now")
    template = TemplateSummariser().summarise(ev)
    say("Template (deterministic, verbalises the evidence pack only):")
    say(template, indent=4, pause=2.5)
    say("")
    say("Local LLM: Qwen2.5-1.5B-Instruct on this CPU, greedy decoding, no hosted API.")
    say("Loading the model ...")
    event("llm_load_start")
    llm = LocalLLMSummariser()
    llm.torch.set_num_threads(os.cpu_count() or 1)  # speed only; the text is checked against the stored run below
    event("llm_generate_start")
    say("Generating (tokens appear as they are produced):")
    sys.stdout.write("    ")
    sys.stdout.flush()
    text = generate_streaming(llm, ev)
    event("llm_generate_end")
    time.sleep(1.0)

    head("[4/4] Faithfulness check (checker v2, run now)")
    for name, txt in (("Template", template), ("Local LLM", text)):
        r = check_summary(txt, ev, name)
        bad = [c for c in r.claims if not c.supported]
        verdict = "all claims supported" if not bad else "flagged: " + "; ".join(f"{c.kind} ({c.detail})" for c in bad)
        say(f"{name}: {len(r.claims)} claims checked, {verdict}", pause=0.8)
    same = text == a["summaries"][gen]["text"]
    say(f"LLM text identical to the stored Week 3 run output: {'yes' if same else 'NO'}", pause=0.8)
    rv = review["alerts"][args.alert]
    say(f"Evidence review of this LLM summary (AI-assisted): {'faithful' if rv['faithful'] else 'NOT faithful'}. "
        f"{rv['reason_en']}", pause=0.5)
    say("Operator-facing alerts use the template summary.", pause=4.0)
    event("end")
    args.events.parent.mkdir(parents=True, exist_ok=True)
    args.events.write_text(json.dumps({"alert": args.alert, "llm_reproduced": same, "events": EVENTS}, indent=1),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
