"""Build the faithfulness review pack ``reports/w03/faithfulness_review_batch.md``.

Read-only over the experiment outputs: for each selected alert it lists the full
evidence pack, the exact prompt and LLM text, the checker v1 flags (from the
original explain run's MLflow artifact) and every checker v2 claim, plus Claude's
earlier read as reference only. This pack was the input of the AI-assisted
review of A04-A20 (verdicts in ``reports/w03/llm_faithfulness_review.json``).

    python scripts/build_w03_review_pack.py            # A04-A20
    python scripts/build_w03_review_pack.py A01 A02    # any alerts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ingen_fleet_health.experiments.explain_run import evidence_from_dict
from ingen_fleet_health.explain.alert_summaries import build_prompt
from ingen_fleet_health.explain.faithfulness import CHECKER_VERSION, check_summary

R = Path("reports/w03")
V1_ARTIFACT = Path("mlruns/w03_explanations/b01f56c899314e2c94a1b12a3766eb2b/artifacts/alert_explanations.json")
G = "local_llm[Qwen/Qwen2.5-1.5B-Instruct]"

cur = json.loads((R / "alert_explanations.json").read_text(encoding="utf-8"))
v1 = json.loads(V1_ARTIFACT.read_text(encoding="utf-8"))
claude = json.loads((R / "manual_faithfulness.json").read_text(encoding="utf-8"))
review = json.loads((R / "llm_faithfulness_review.json").read_text(encoding="utf-8"))

assert CHECKER_VERSION == 2 == cur["checker_version"]
wanted = sys.argv[1:] or [f"A{i:02d}" for i in range(4, 21)]
todo = [a for a in cur["alerts"] if a["alert_id"] in wanted]
v1_by_id = {a["alert_id"]: a for a in v1["alerts"]}


def num(x: float | None) -> str:
    if x is None:
        return "无有效读数（缺失）"
    if x == 0:
        return "0"
    return f"{x:.4g}"


def cell(s: str) -> str:
    return "`" + s.replace("|", "\\|").replace("`", "'") + "`"


def tcell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


out: list[str] = []
w = out.append
sys_prompt = build_prompt(evidence_from_dict(cur["alerts"][0]["evidence"]))[0]

w("# Week 3 LLM 告警说明：忠实度审核材料包")
w("")
w("> **全部为合成数据**（模拟器 v0.2 的 Aido Rover 车队与合成维护记录语料），不含真实客户数据或公司内部资料。")
w("> 本文件只汇编已有实验输出，不包含审核结论（结论见 `reports/w03/llm_faithfulness_review.json`）。第 5 节是 Claude 此前的判读，**仅供参考，不是审核结论**。")
w("")
w("## 范围")
w("")
w(f"- 本文件收录 {len(todo)} 条本地 LLM 告警说明：{', '.join(a['alert_id'] for a in todo)}。"
  "A04–A20 的 AI 辅助审核以本材料包为输入；A01–A03 在对话中逐条审核。")
w(f"- 被审核的生成器：`{G}`，在本机 CPU 上运行，未调用任何托管 LLM API。")
w("")
w("## 数据来源（均未修改）")
w("")
w(f"- 证据包、prompt、LLM 原文、检查器 v2 结果：`reports/w03/alert_explanations.json`"
  f"（解释实验 MLflow run `{cur['mlflow_run_id']}`；v2 重新打分 run `{cur['recheck_mlflow_run_id']}`）。")
w(f"- 检查器 v1 结果：原始实验 run 的 MLflow 产物 `{V1_ARTIFACT.as_posix()}`。"
  "其中 LLM 原文、prompt 与证据包与当前文件逐字节一致（已校验）。")
w("- Claude 此前的判读：`reports/w03/manual_faithfulness.json`。")
w("- 被解释的模型：" + cur["explained_model"] + "。SHAP：" + cur["shap"]["method"] + "。")
w("")
w("## 审核标准（A01–A20 相同）")
w("")
w("> " + review["criterion"])
w("")
w("## 如何阅读每条案例")
w("")
w("- **SHAP 方向**：正值 = 该特征把风险往上推；负值 = 往下拉。SHAP 是模型对输出的贡献，不等于现实中某件事正在发生。")
w("- **风险与 SHAP 的关系**：模型风险是校准后的概率（每折各自校准）；SHAP、基准值和原始 margin 来自未校准模型的 log-odds。"
  "两者排序一致，但数值不能直接换算（例如 margin 很低时，校准后的风险仍可能是几个百分点）。")
w("- **证据包**：每条告警取 SHAP 最大的 5 个正向特征和 2 个负向特征，各给出当前值与 72 h 前的值，以及告警前 7 天的关联记录（操作员记录与服务工单）。")
w("- **LLM 实际看到的只是证据包的一部分**：前 3 个正向特征、第 1 个负向特征、最新的 2 条操作员记录。"
  "数值四舍五入到 3 位有效数字；若 72 h 前的值在该精度下与当前值相同，prompt 中省略 72 h 前的值。"
  "证据表中“进入 prompt”一列标出哪些内容 LLM 看得到。")
w("- **LLM 设置**：贪婪解码（do_sample=False），repetition_penalty=1.05，max_new_tokens=160，"
  "因此部分输出在句中被截断。存储的原文是模型解码输出把连续空白合并为单个空格后的结果；本文件逐字照录，未做其他修改。")
w("- **自动检查器**是基于词表和正则的检查，抽取 6 类声明：quote（引文须来自关联记录）、feature（原样写出的特征名及其方向）、"
  "reading（分类器判读须出现在关联记录中）、number（数字须在写出精度下等于证据包中某个值）、"
  "cause（自由措辞中的原因概念须在正向特征或关联记录中）、notes（关于操作员记录的说法）。"
  "它不检查数字属于哪个特征或哪个时间点，也不检查“上升/下降”“strong”等趋势和程度措辞。"
  "v1 为原始实验时的版本；v2 修正了 5 类通用误报原因后，对同一批存储文本重新打分（未重新生成）。")
w("- **v1 只保存了未通过的声明**（以及声明总数）；v2 列出全部声明。")
w("")
w("## 所有案例共用的系统指令（原样）")
w("")
w("```text")
w(sys_prompt)
w("```")
w("")

for a in todo:
    aid = a["alert_id"]
    e = a["evidence"]
    ev = evidence_from_dict(e)
    prompt = a["prompt"]
    s = a["summaries"][G]
    text = s["text"]
    assert "```" not in text and "```" not in prompt
    sys_p, user_p = build_prompt(ev)
    assert user_p == prompt and sys_p == sys_prompt, aid
    o = v1_by_id[aid]
    assert o["summaries"][G]["text"] == text and o["prompt"] == prompt and o["evidence"] == e, aid
    v1c = o["summaries"][G]["check"]
    assert v1c["faithful"] == cur["checker_history"][0]["faithful_per_alert"][aid][G], aid
    v2c = s["check"]
    r = check_summary(text, ev, G)
    assert r.faithful == v2c["faithful"] and len(r.claims) == v2c["claims"], aid
    assert [c.text for c in r.claims if not c.supported] == [u["text"] for u in v2c["unsupported"]], aid

    w("---")
    w("")
    w(f"## {aid} · {e['unit_id']} · {e['t']} UTC")
    w("")
    w("### 1. 案例信息")
    w("")
    w(f"- 模型风险：{100 * e['risk']:.2f} %（prompt 中写作 {prompt.split('72 h: ')[1].split(' %')[0]} %）")
    w(f"- 告警阈值：{100 * e['threshold']:.2f} %（prompt 中写作 {prompt.split('Alert threshold: ')[1].split(' %')[0]} %）")
    w(f"- 实际结果：`{a['outcome']}`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）")
    w(f"- 折：`{e['fold_id']}`；模型基准值（log-odds）{e['base_value']:.3f}；本条原始 margin {e['raw_margin']:.3f}")
    w("")
    w("### 2. 证据包（完整）与 LLM 收到的 prompt")
    w("")
    for title, rows in (("推高风险的特征（SHAP > 0，按贡献从大到小）", e["positive"]),
                        ("拉低风险的特征（SHAP < 0，按贡献从大到小）", e["negative"])):
        w(f"**{title}**")
        w("")
        w("| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k, f in enumerate(rows, 1):
            inp = "是" if f["phrase"] in prompt else "否"
            dirn = "推高" if f["shap"] > 0 else "拉低"
            w(f"| {k} | `{f['feature']}` | {tcell(f['phrase'])} | {num(f['value'])} | {num(f['value_72h_ago'])} | "
              f"{f['unit'] or '—'} | {f['shap']:+.4f} | {dirn} | {inp} |")
        if not rows:
            w("| — | 无 | | | | | | | |")
        w("")
    w("**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）")
    w("")
    if e["tickets"]:
        w("| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |")
        w("|---|---|---|---|---|---|---|")
        for t in e["tickets"]:
            inp = "是" if f"\"{t['text']}\"" in prompt else "否"
            w(f"| {t['ticket_id']} | {t['doc_type']} | {t['age_hours']:.2f} | {cell(t['text'])} | "
              f"{t['predicted_class']} | {t['predicted_probability']:.3f} | {inp} |")
    else:
        w("无任何关联的操作员记录或服务工单。")
    w("")
    w("**LLM 收到的 user prompt（原样）**")
    w("")
    w("```text")
    w(prompt)
    w("```")
    w("")
    w("### 3. LLM 原始输出（原样，未修改、未截断、未意译）")
    w("")
    w("```text")
    w(text)
    w("```")
    w("")
    w(f"句数（按存储的切分）：{s['sentences']}；系统指令要求 2–3 句。")
    w("")
    w("### 4. 自动检查器结论")
    w("")
    w(f"**v1**（原始实验）：{'忠实' if v1c['faithful'] else '不忠实'}；抽取声明 {v1c['claims']} 条，"
      f"未通过 {len(v1c['unsupported'])} 条。")
    w("")
    if v1c["unsupported"]:
        w("| 类型 | 检查器给出的原因 | 被检查的文本 |")
        w("|---|---|---|")
        for u in v1c["unsupported"]:
            w(f"| {u['kind']} | {tcell(u['detail'])} | {tcell(u['text'])} |")
        w("")
    w(f"**v2**（重新打分）：{'忠实' if v2c['faithful'] else '不忠实'}；抽取声明 {v2c['claims']} 条，"
      f"未通过 {len(v2c['unsupported'])} 条。全部声明：")
    w("")
    if r.claims:
        w("| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |")
        w("|---|---|---|---|---|")
        for k, c in enumerate(r.claims, 1):
            w(f"| {k} | {c.kind} | {'是' if c.supported else '**否**'} | {tcell(c.detail)} | {tcell(c.text)} |")
    else:
        w("未抽取到任何声明。")
    w("")
    w("### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）")
    w("")
    cl = claude[G][aid]
    w(f"- Claude 判读：{'忠实' if cl['faithful'] else '不忠实'}")
    afv = cl.get("auto_flag_valid")
    w(f"- Claude 对检查器标记是否有效的看法（auto_flag_valid）：{'null（检查器未标记）' if afv is None else str(afv).lower()}")
    w(f"- Claude 的说明（原文）：{cl['note']}")
    w("")
    w("### 审核判定（待审核者填写）")
    w("")
    w("- faithful：")
    w("- 主要原因：")
    w("- 次要问题：")
    w("- 格式问题（单独记录，不作为主要判定依据）：")
    w("- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：")
    w("")

path = R / "faithfulness_review_batch.md"
path.write_text("\n".join(out), encoding="utf-8")
print(path.resolve(), len(todo), "alerts", sum(len(x) for x in out), "chars")
