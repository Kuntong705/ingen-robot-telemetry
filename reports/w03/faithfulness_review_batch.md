# Week 3 LLM 告警说明：忠实度审核材料包

> **全部为合成数据**（模拟器 v0.2 的 Aido Rover 车队与合成维护记录语料），不含真实客户数据或公司内部资料。
> 本文件只汇编已有实验输出，不包含审核结论（结论见 `reports/w03/llm_faithfulness_review.json`）。第 5 节是 Claude 此前的判读，**仅供参考，不是审核结论**。

## 范围

- 本文件收录 17 条本地 LLM 告警说明：A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20。A04–A20 的 AI 辅助审核以本材料包为输入；A01–A03 在对话中逐条审核。
- 被审核的生成器：`local_llm[Qwen/Qwen2.5-1.5B-Instruct]`，在本机 CPU 上运行，未调用任何托管 LLM API。

## 数据来源（均未修改）

- 证据包、prompt、LLM 原文、检查器 v2 结果：`reports/w03/alert_explanations.json`（解释实验 MLflow run `b01f56c899314e2c94a1b12a3766eb2b`；v2 重新打分 run `f89918c62fd14cd29166177b3f313efd`）。
- 检查器 v1 结果：原始实验 run 的 MLflow 产物 `mlruns/w03_explanations/b01f56c899314e2c94a1b12a3766eb2b/artifacts/alert_explanations.json`。其中 LLM 原文、prompt 与证据包与当前文件逐字节一致（已校验）。
- Claude 此前的判读：`reports/w03/manual_faithfulness.json`。
- 被解释的模型：early_fusion (LightGBM on telemetry + text features, variant chosen on validation)。SHAP：TreeExplainer, exact tree SHAP, raw margin (log-odds) of the uncalibrated model。

## 审核标准（A01–A20 相同）

> Every stated fact, number, direction and causal explanation must be supported by the evidence pack. Format violations (more than 3 sentences, truncation) are recorded separately and do not by themselves decide the verdict.

## 如何阅读每条案例

- **SHAP 方向**：正值 = 该特征把风险往上推；负值 = 往下拉。SHAP 是模型对输出的贡献，不等于现实中某件事正在发生。
- **风险与 SHAP 的关系**：模型风险是校准后的概率（每折各自校准）；SHAP、基准值和原始 margin 来自未校准模型的 log-odds。两者排序一致，但数值不能直接换算（例如 margin 很低时，校准后的风险仍可能是几个百分点）。
- **证据包**：每条告警取 SHAP 最大的 5 个正向特征和 2 个负向特征，各给出当前值与 72 h 前的值，以及告警前 7 天的关联记录（操作员记录与服务工单）。
- **LLM 实际看到的只是证据包的一部分**：前 3 个正向特征、第 1 个负向特征、最新的 2 条操作员记录。数值四舍五入到 3 位有效数字；若 72 h 前的值在该精度下与当前值相同，prompt 中省略 72 h 前的值。证据表中“进入 prompt”一列标出哪些内容 LLM 看得到。
- **LLM 设置**：贪婪解码（do_sample=False），repetition_penalty=1.05，max_new_tokens=160，因此部分输出在句中被截断。存储的原文是模型解码输出把连续空白合并为单个空格后的结果；本文件逐字照录，未做其他修改。
- **自动检查器**是基于词表和正则的检查，抽取 6 类声明：quote（引文须来自关联记录）、feature（原样写出的特征名及其方向）、reading（分类器判读须出现在关联记录中）、number（数字须在写出精度下等于证据包中某个值）、cause（自由措辞中的原因概念须在正向特征或关联记录中）、notes（关于操作员记录的说法）。它不检查数字属于哪个特征或哪个时间点，也不检查“上升/下降”“strong”等趋势和程度措辞。v1 为原始实验时的版本；v2 修正了 5 类通用误报原因后，对同一批存储文本重新打分（未重新生成）。
- **v1 只保存了未通过的声明**（以及声明总数）；v2 列出全部声明。

## 所有案例共用的系统指令（原样）

```text
You write short alert explanations for robot fleet operators. Use ONLY the facts listed under EVIDENCE. Do not add causes, numbers, advice or component names that are not in EVIDENCE. Copy numbers exactly as written. Write 2 to 3 plain English sentences in one paragraph, and nothing else.
```

---

## A04 · Rover_002 · 2026-12-04 12:00 UTC

### 1. 案例信息

- 模型风险：2.84 %（prompt 中写作 2.8 %）
- 告警阈值：2.29 %（prompt 中写作 2.3 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1125_g0`；模型基准值（log-odds）-10.733；本条原始 margin -14.013

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `terrain_patrol_mean_24h` | mean terrain load while active (24 h) | 0.4331 | 0.3995 | — | +0.1045 | 推高 | 是 |
| 2 | `vib_norm_mean_6h` | terrain-normalised vibration (6 h mean) | 0.1744 | 0.1739 | g | +0.0936 | 推高 | 是 |
| 3 | `txt_service_count_7d` | service tickets in the last 7 days | 0 | 0 | — | +0.0419 | 推高 | 是 |
| 4 | `txt_decay_thermal` | recent operator notes read as 'thermal' (time-decayed weight) | 0.0004527 | 0.0007749 | — | +0.0224 | 推高 | 否 |
| 5 | `r_hat_temp_norm_24h` | temperature-corrected battery internal resistance (24 h) | 无有效读数（缺失） | 0.04826 | ohm | +0.0164 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `motor_temp_resid_dock_mean_24h` | docked motor-temperature offset from ambient (24 h mean) | 0.091 | 0.09875 | deg C | -0.7191 | 拉低 | 是 |
| 2 | `txt_decay_user_or_operational_error` | recent operator notes read as 'user or operational error' (time-decayed weight) | 0.9437 | 0.0001152 | — | -0.5336 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00856 | operator_note | 3.85 | `Pre-shift check was skipped & the patrpl started late` | user_or_operational_error | 0.995 | 是 |
| T00848 | operator_note | 16.82 | `heavy rain today, patrol completed as planned, reported by the guard on site` | no_fault_found | 0.996 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_002. Alert time: 2026-12-04 12:00 UTC.
Model risk of a qualifying fault onset within 72 h: 2.8 %. Alert threshold: 2.3 %.
Features pushing the risk UP (strongest first):
- mean terrain load while active (24 h) at 0.433 (0.400 72 h earlier)
- terrain-normalised vibration (6 h mean) at 0.174 g
- service tickets in the last 7 days at 0
Features pushing the risk DOWN:
- docked motor-temperature offset from ambient (24 h mean) at 0.0910 deg C (0.0987 deg C 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 3.85 h before the alert: "Pre-shift check was skipped & the patrpl started late" (ticket classifier reading: user or operational error)
- 16.8 h before the alert: "heavy rain today, patrol completed as planned, reported by the guard on site" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_002 is experiencing a higher-than-expected risk of a qualifying fault onset within the next 72 hours due to several factors. The mean terrain load while active has increased by 0.433 units compared to 72 hours ago, indicating potential wear on the rover's components. Additionally, the terrain-normalized vibration level has risen to 0.174 g, suggesting increased stress on the vehicle's structure. Furthermore, there have been no service tickets filed in the past seven days, which could indicate a lack of proactive maintenance. However, there are mitigating factors that may reduce the risk. The docked motor-temperature offset from ambient has decreased slightly, indicating better thermal management. The patrol was completed as planned despite heavy rain, suggesting that the
```

句数（按存储的切分）：7；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 8 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| notes | 2 linked operator note(s) exist | Furthermore, there have been no service tickets filed in the past seven days, which could indicate a lack of proactive maintenance. |

**v2**（重新打分）：不忠实；抽取声明 7 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.433 |
| 2 | number | 是 | equals a pack value at the written precision | 0.174 |
| 3 | cause | 是 | concept is in the attributions / linked tickets | environment: The mean terrain load |
| 4 | cause | 是 | concept is in the attributions / linked tickets | vibration: Additionally, the terrain-normalised vibration level has risen to 0.174 g, suggesting increased stress on the vehicle's structure. |
| 5 | notes | **否** | 2 linked operator note(s) exist | Furthermore, there have been no service tickets filed in the past seven days, which could indicate a lack of proactive maintenance. |
| 6 | cause | 是 | concept is in the attributions / linked tickets | thermal: The docked motor-temperature offset from ambient has decreased slightly, indicating better thermal management. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | environment: The docked motor-temperature offset from ambient has decreased slightly, indicating better thermal management. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：false
- Claude 的说明（原文）：'Increased by 0.433 units' (0.400 to 0.433 is +0.034); invents 'wear on components', 'lack of proactive maintenance', 'better thermal management'. Checker flag (no service tickets) is a false flag: there are none.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A05 · Rover_046 · 2026-12-05 00:00 UTC

### 1. 案例信息

- 模型风险：85.65 %（prompt 中写作 86 %）
- 告警阈值：12.50 %（prompt 中写作 12 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g3`；模型基准值（log-odds）-10.403；本条原始 margin 0.739

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `vib_norm_ratio_to_baseline` | terrain-normalised vibration relative to the unit's own baseline | 1.466 | 无有效读数（缺失） | x baseline | +6.1781 | 推高 | 是 |
| 2 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.014 | 无有效读数（缺失） | x baseline | +4.8066 | 推高 | 是 |
| 3 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | 1.67 | 无有效读数（缺失） | deg C | +0.6217 | 推高 | 是 |
| 4 | `txt_note_count_72h` | operator notes in the last 72 h | 3 | 0 | — | +0.4424 | 推高 | 否 |
| 5 | `vib_norm_mean_6h` | terrain-normalised vibration (6 h mean) | 0.2211 | 无有效读数（缺失） | g | +0.4406 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.0006397 | 0.001599 | — | -0.4273 | 拉低 | 是 |
| 2 | `vib_norm_change_72h` | change in terrain-normalised vibration over 72 h | 无有效读数（缺失） | 无有效读数（缺失） | g | -0.3813 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00832 | operator_note | 38.53 | `Weekly visual insp done no issues` | no_fault_found | 0.994 | 是 |
| T00823 | operator_note | 55.37 | `Rover 46 - normal patrol, no incidents. loghing for maint` | no_fault_found | 0.995 | 是 |
| T00808 | operator_note | 64.23 | `charger was unplugged by cleaning staff unit did not chg overnight same as last shidt` | user_or_operational_error | 0.998 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_046. Alert time: 2026-12-05 00:00 UTC.
Model risk of a qualifying fault onset within 72 h: 86 %. Alert threshold: 12 %.
Features pushing the risk UP (strongest first):
- terrain-normalised vibration relative to the unit's own baseline at 1.47 x baseline
- terrain-normalised energy use relative to the unit's own baseline at 1.01 x baseline, i.e. essentially at the reference level
- motor temperature above the load/ambient model while active (24 h mean) at 1.67 deg C
Features pushing the risk DOWN:
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.000640 (0.00160 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 38.5 h before the alert: "Weekly visual insp done no issues" (ticket classifier reading: no fault found)
- 55.4 h before the alert: "Rover 46 - normal patrol, no incidents. loghing for maint" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
The rover 46 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to several factors. The strongest indicators are: - A significant increase in terrain-normalized vibration relative to the unit's baseline, reaching 1.47 times the baseline value. - An elevated energy use compared to the unit's baseline, with a factor of 1.01, indicating near-normal operation. - A motor temperature above the load/ambient model during active operation, at 1.67 degrees Celsius, which is above the threshold. However, there are also mitigating factors: - Recent operator notes indicate routine maintenance activities and no reported faults, with weights decaying over time. - The most recent note from 38.5 hours prior states that a
```

句数（按存储的切分）：6；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 12 条，未通过 4 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| number | no pack value rounds to this number | 46 |
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: The strongest indicators are: - A significant increase in terrain-normalized vibration relative to the unit's baseline, reaching 1.47 times the baseline value. |
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: - A motor temperature above the load/ambient model during active operation, at 1.67 degrees Celsius, which is above the threshold. |
| notes | 3 linked operator note(s) exist | However, there are also mitigating factors: - Recent operator notes indicate routine maintenance activities and no reported faults, with weights decaying over time. |

**v2**（重新打分）：不忠实；抽取声明 10 条，未通过 2 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 1.47 |
| 2 | number | 是 | equals a pack value at the written precision | 1.01 |
| 3 | number | 是 | equals a pack value at the written precision | 1.67 |
| 4 | number | 是 | equals a pack value at the written precision | 38.5 |
| 5 | cause | 是 | concept is in the attributions / linked tickets | vibration: The strongest indicators are: - A significant increase in terrain-normalised vibration relative to the unit's baseline, reaching 1.47 times the baseline value. |
| 6 | cause | 是 | concept is in the attributions / linked tickets | energy: - An elevated energy use compared to the unit's baseline, with a factor of 1.01, indicating near-normal operation. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | thermal: - A motor temperature above the load/ambient model during active operation, at 1.67 degrees Celsius, which is above the threshold. |
| 8 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | environment: - A motor temperature above the load/ambient model during active operation, at 1.67 degrees Celsius, which is above the threshold. |
| 9 | notes | **否** | 3 linked operator note(s) exist | However, there are also mitigating factors: - Recent operator notes indicate routine maintenance activities and no reported faults, with weights decaying over time. |
| 10 | notes | 是 | linked ticket(s) exist | - The most recent note from 38.5 hours prior states that a |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：false
- Claude 的说明（原文）：Invents 'above the threshold' for the motor-temperature excess; truncated. Checker flags are spurious ('ambient' inside a feature definition; negation).

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A06 · Rover_018 · 2026-12-05 08:00 UTC

### 1. 案例信息

- 模型风险：11.84 %（prompt 中写作 12 %）
- 告警阈值：11.84 %（prompt 中写作 12 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g0`；模型基准值（log-odds）-10.655；本条原始 margin -3.543

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0.6714 | 0.0009975 | — | +1.7208 | 推高 | 是 |
| 2 | `r_bms_max_24h` | 24 h maximum of BMS-reported internal resistance | 0.0669 | 0.05266 | ohm | +1.2116 | 推高 | 是 |
| 3 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.06193 | 0.0004446 | — | +1.1115 | 推高 | 是 |
| 4 | `soc_min_24h` | 24 h minimum of battery state of charge | 94.92 | 36.84 | % | +0.6147 | 推高 | 否 |
| 5 | `vib_norm_mean_24h` | terrain-normalised vibration (24 h mean) | 无有效读数（缺失） | 0.1572 | g | +0.5846 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 0.03835 | 0.3205 | — | -0.8556 | 拉低 | 是 |
| 2 | `txt_decay_sensor` | recent operator notes read as 'sensor' (time-decayed weight) | 0.257 | 0.5332 | — | -0.4235 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00837 | operator_note | 43.12 | `Rover_018 - SoH on the display went down again, not urgent` | battery | 0.542 | 是 |
| T00778 | operator_note | 115.37 | `Proximity sensor on the rear right side gives no reading reported by the guard on site` | sensor | 0.996 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_018. Alert time: 2026-12-05 08:00 UTC.
Model risk of a qualifying fault onset within 72 h: 12 %. Alert threshold: 12 %.
Features pushing the risk UP (strongest first):
- strongest fault wording in operator notes of the last 72 h (classifier probability) at 0.671 (0.000998 72 h earlier)
- 24 h maximum of BMS-reported internal resistance at 0.0669 ohm (0.0527 ohm 72 h earlier)
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.0619 (0.000445 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'no fault found' (time-decayed weight) at 0.0383 (0.321 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 43.1 h before the alert: "Rover_018 - SoH on the display went down again, not urgent" (ticket classifier reading: battery)
- 115 h before the alert: "Proximity sensor on the rear right side gives no reading reported by the guard on site" (ticket classifier reading: sensor)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
The unit Rover_018 is at an elevated risk of experiencing a qualifying fault within the next 72 hours due to several factors. The most significant indicators are: - A strong fault was noted in the operator's notes from 72 hours ago, with a classifier probability of 0.671, suggesting a high likelihood of a fault occurring. - The BMS-reported internal resistance has been consistently low at 0.0669 ohms over the past 72 hours, compared to 0.0527 ohms previously, indicating potential issues with the vehicle's electrical system. - Recent operator notes mentioning "drivetrain" have a time-decayed weight of 0.0619, which suggests ongoing concern about drivetrain-related
```

句数（按存储的切分）：4；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 9 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| quote | no linked ticket has this text | drivetrain |

**v2**（重新打分）：忠实；抽取声明 8 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.671 |
| 2 | number | 是 | equals a pack value at the written precision | 0.0669 |
| 3 | number | 是 | equals a pack value at the written precision | 0.0527 |
| 4 | number | 是 | equals a pack value at the written precision | 0.0619 |
| 5 | notes | 是 | linked ticket(s) exist | The most significant indicators are: - A strong fault was noted in the operator's notes from 72 hours ago, with a classifier probability of 0.671, suggesting a high likelihood of a fault occurring. |
| 6 | cause | 是 | concept is in the attributions / linked tickets | battery: - The BMS-reported internal resistance has been consistently low at 0.0669 ohms over the past 72 hours, compared to 0.0527 ohms previously, indicating potential issues with the vehicle's electrical system. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | vibration: - Recent operator notes mentioning  drivetrain  have a time-decayed weight of 0.0619, which suggests ongoing concern about drivetrain-related |
| 8 | notes | 是 | linked ticket(s) exist | - Recent operator notes mentioning  drivetrain  have a time-decayed weight of 0.0619, which suggests ongoing concern about drivetrain-related |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：Says the fault note is 'from 72 hours ago' (it is within the last 72 h; 72 h ago the value was 0.001); calls a rising resistance 'consistently low'.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A07 · Rover_026 · 2026-12-05 19:00 UTC

### 1. 案例信息

- 模型风险：4.91 %（prompt 中写作 4.9 %）
- 告警阈值：4.91 %（prompt 中写作 4.9 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g2`；模型基准值（log-odds）-10.349；本条原始 margin -7.098

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.001 | 1.001 | x baseline | +1.2065 | 推高 | 是 |
| 2 | `r_hat_temp_norm_24h` | temperature-corrected battery internal resistance (24 h) | 0.04265 | 0.04443 | ohm | +0.4530 | 推高 | 是 |
| 3 | `energy_ratio_to_expected_24h` | energy use relative to what terrain and ambient predict (24 h) | 0.9516 | 0.9515 | x expected | +0.4501 | 推高 | 是 |
| 4 | `motor_temp_resid_dock_mean_24h` | docked motor-temperature offset from ambient (24 h mean) | 0.4962 | 0.3046 | deg C | +0.4293 | 推高 | 否 |
| 5 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | 0.5958 | -0.3913 | deg C | +0.4011 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.000304 | 0.0002595 | — | -0.9018 | 拉低 | 是 |
| 2 | `txt_decay_connectivity` | recent operator notes read as 'connectivity' (time-decayed weight) | 0.01234 | 0.000652 | — | -0.3036 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00890 | operator_note | 0.58 | `Connection stable no dropoyts` | no_fault_found | 0.986 | 是 |
| T00825 | operator_note | 72.75 | `rover_026 - visitors ased about the robot, no issues` | no_fault_found | 0.998 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_026. Alert time: 2026-12-05 19:00 UTC.
Model risk of a qualifying fault onset within 72 h: 4.9 %. Alert threshold: 4.9 %.
Features pushing the risk UP (strongest first):
- terrain-normalised energy use relative to the unit's own baseline at 1.00 x baseline, i.e. essentially at the reference level
- temperature-corrected battery internal resistance (24 h) at 0.0426 ohm (0.0444 ohm 72 h earlier)
- energy use relative to what terrain and ambient predict (24 h) at 0.952 x expected (0.951 x expected 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.000304 (0.000260 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 0.583 h before the alert: "Connection stable no dropoyts" (ticket classifier reading: no fault found)
- 72.8 h before the alert: "rover_026 - visitors ased about the robot, no issues" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_026 is currently experiencing a low risk of a qualifying fault onset within the next 72 hours, with a risk of 4.9%. The risk is being pushed up by several factors, including high terrain-normalized energy use compared to its baseline, a slightly higher than expected battery internal resistance, and an energy use rate that exceeds expectations based on terrain and ambient conditions. However, the risk is also being mitigated by recent operator notes indicating stable connections and no reported issues.
```

句数（按存储的切分）：3；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 5 条，未通过 2 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: The risk is being pushed up by several factors, including high terrain-normalized energy use compared to its baseline, a slightly higher than expected battery internal resistance, and an energy use rate that exceeds expectations based on terrain and ambient conditions. |
| notes | 2 linked operator note(s) exist | However, the risk is also being mitigated by recent operator notes indicating stable connections and no reported issues. |

**v2**（重新打分）：不忠实；抽取声明 5 条，未通过 2 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 4.9 |
| 2 | cause | 是 | concept is in the attributions / linked tickets | battery: The risk is being pushed up by several factors, including high terrain-normalised energy use compared to its baseline, a slightly higher than expected battery internal resistance, and an energy use rate that exceeds expectations based on terrain and ambient conditions. |
| 3 | cause | 是 | concept is in the attributions / linked tickets | energy: The risk is being pushed up by several factors, including high terrain-normalised energy use compared to its baseline, a slightly higher than expected battery internal resistance, and an energy use rate that exceeds expectations based on terrain and ambient conditions. |
| 4 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | environment: The risk is being pushed up by several factors, including high terrain-normalised energy use compared to its baseline, a slightly higher than expected battery internal resistance, and an energy use rate that exceeds expectations based on terrain and ambient conditions. |
| 5 | notes | **否** | 2 linked operator note(s) exist | However, the risk is also being mitigated by recent operator notes indicating stable connections and no reported issues. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：false
- Claude 的说明（原文）：Says energy use 'exceeds expectations' at 0.952 x expected and calls a 1.0009 x baseline value 'high'; says resistance is higher than expected while it fell. Checker flags are spurious.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A08 · Rover_001 · 2026-12-07 20:00 UTC

### 1. 案例信息

- 模型风险：16.03 %（prompt 中写作 16 %）
- 告警阈值：16.03 %（prompt 中写作 16 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g4`；模型基准值（log-odds）-10.839；本条原始 margin -4.624

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.1962 | 0.001787 | — | +2.7376 | 推高 | 是 |
| 2 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.002 | 无有效读数（缺失） | x baseline | +1.1832 | 推高 | 是 |
| 3 | `txt_decay_connectivity` | recent operator notes read as 'connectivity' (time-decayed weight) | 0.05649 | 0.1207 | — | +0.8523 | 推高 | 是 |
| 4 | `r_hat_temp_norm_24h` | temperature-corrected battery internal resistance (24 h) | 0.0483 | 无有效读数（缺失） | ohm | +0.7693 | 推高 | 否 |
| 5 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0.4422 | 0.1494 | — | +0.6672 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | 1.493 | 无有效读数（缺失） | deg C | -1.1744 | 拉低 | 是 |
| 2 | `txt_service_count_7d` | service tickets in the last 7 days | 2 | 0 | — | -0.5957 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00924 | operator_note | 9.65 | `Evening shift batt door seal worn moisture warning - same as last shift` | no_fault_found | 0.954 | 是 |
| T00913 | service_ticket | 25.82 | `Replaced front right wheel brg. Tets drive ok, vibratin back to nrml.` | drivetrain | 0.960 | 否 |
| T00891 | service_ticket | 49.35 | `Wok order opened: vibration alarm on drivetrain. Check front right bearing plya.` | no_fault_found | 0.587 | 否 |
| T00886 | operator_note | 51.20 | `Note for maintenance not running right today not urgent` | no_fault_found | 0.954 | 是 |
| T00880 | operator_note | 55.17 | `rattling spund from the drive unit at low speed needs a look asap` | no_fault_found | 0.505 | 否 |
| T00862 | operator_note | 77.83 | `Unit 1: comes back to dock earlier than it used to` | no_fault_found | 0.576 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_001. Alert time: 2026-12-07 20:00 UTC.
Model risk of a qualifying fault onset within 72 h: 16 %. Alert threshold: 16 %.
Features pushing the risk UP (strongest first):
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.196 (0.00179 72 h earlier)
- terrain-normalised energy use relative to the unit's own baseline at 1.00 x baseline, i.e. essentially at the reference level
- recent operator notes read as 'connectivity' (time-decayed weight) at 0.0565 (0.121 72 h earlier)
Features pushing the risk DOWN:
- motor temperature above the load/ambient model while active (24 h mean) at 1.49 deg C
Operator notes for this unit in the previous 7 days (newest first):
- 9.65 h before the alert: "Evening shift batt door seal worn moisture warning - same as last shift" (ticket classifier reading: no fault found)
- 51.2 h before the alert: "Note for maintenance not running right today not urgent" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_001 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to several factors. The most significant contributors are recent operator notes indicating drivetrain issues (weight factor 0.196, decaying over 72 hours), which have increased by 16% since the last assessment. Additionally, the unit's energy usage has remained relatively stable compared to its baseline, contributing only a slight increase (weight factor 0.0565). However, there are also indicators of potential problems, such as motor temperatures being slightly above normal levels (weight factor 1.49, decaying over 24 hours), which could be a cause for concern. Despite these risks, the unit's recent operator notes suggest that any
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 9 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | thermal: However, there are also indicators of potential problems, such as motor temperatures being slightly above normal levels (weight factor 1.49, decaying over 24 hours), which could be a cause for concern. |

**v2**（重新打分）：不忠实；抽取声明 9 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.196 |
| 2 | number | 是 | equals a pack value at the written precision | 16 |
| 3 | number | 是 | equals a pack value at the written precision | 0.0565 |
| 4 | number | 是 | equals a pack value at the written precision | 1.49 |
| 5 | cause | 是 | concept is in the attributions / linked tickets | vibration: The most significant contributors are recent operator notes indicating drivetrain issues (weight factor 0.196, decaying over 72 hours), which have increased by 16% since the last assessment. |
| 6 | notes | 是 | a note feature is a positive attribution | The most significant contributors are recent operator notes indicating drivetrain issues (weight factor 0.196, decaying over 72 hours), which have increased by 16% since the last assessment. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | energy: Additionally, the unit's energy usage has remained relatively stable compared to its baseline, contributing only a slight increase (weight factor 0.0565). |
| 8 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | thermal: However, there are also indicators of potential problems, such as motor temperatures being slightly above normal levels (weight factor 1.49, decaying over 24 hours), which could be a cause for concern. |
| 9 | notes | 是 | linked ticket(s) exist | Despite these risks, the unit's recent operator notes suggest that any |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：Presents the motor-temperature excess (a risk-lowering attribution) as a concern; invents '+16 %'; attaches the connectivity-note value 0.0565 to energy use.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A09 · Rover_050 · 2026-12-08 12:00 UTC

### 1. 案例信息

- 模型风险：4.91 %（prompt 中写作 4.9 %）
- 告警阈值：4.91 %（prompt 中写作 4.9 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g2`；模型基准值（log-odds）-10.349；本条原始 margin -5.923

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.21 | 1.115 | x baseline | +2.4773 | 推高 | 是 |
| 2 | `r_hat_temp_norm_24h` | temperature-corrected battery internal resistance (24 h) | 0.04856 | 0.04473 | ohm | +1.0918 | 推高 | 是 |
| 3 | `txt_decay_thermal` | recent operator notes read as 'thermal' (time-decayed weight) | 8.08e-05 | 0.0002285 | — | +0.9735 | 推高 | 是 |
| 4 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 0.0002084 | 0.0005895 | — | +0.8394 | 推高 | 否 |
| 5 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | 0.05507 | -0.00204 | deg C | +0.4086 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.0001884 | 0.0005328 | — | -1.2193 | 拉低 | 是 |
| 2 | `hours_since_repair` | hours since the last completed repair | 684 | 612 | h | -0.2568 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00799 | operator_note | 162.62 | `sent off the path onto wet grass wheels spinning had to be pushed - same as last shift` | user_or_operational_error | 0.994 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_050. Alert time: 2026-12-08 12:00 UTC.
Model risk of a qualifying fault onset within 72 h: 4.9 %. Alert threshold: 4.9 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.21 x baseline (1.11 x baseline 72 h earlier)
- temperature-corrected battery internal resistance (24 h) at 0.0486 ohm (0.0447 ohm 72 h earlier)
- recent operator notes read as 'thermal' (time-decayed weight) at 0.000081 (0.000229 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.000188 (0.000533 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 163 h before the alert: "sent off the path onto wet grass wheels spinning had to be pushed - same as last shift" (ticket classifier reading: user or operational error)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_050 is at risk of a qualifying fault onset within 72 hours due to an increased battery internal resistance compared to its baseline, which has risen from 0.0447 ohm to 0.0486 ohm over the past 24 hours. The temperature-corrected battery internal resistance also shows a slight increase from 0.0447 ohm to 0.0486 ohm, indicating potential thermal issues. Recent operator notes mentioning 'thermal' issues have a decaying weight of 0.000081, while 'drivetrain' issues have a decaying weight of 0.000188, suggesting these may be more relevant to the current condition.
```

句数（按存储的切分）：3；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 12 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | vibration: 'drivetrain' issues have a decaying weight of 0.000188, suggesting these may be more relevant to the current condition. |

**v2**（重新打分）：不忠实；抽取声明 12 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.0447 |
| 2 | number | 是 | equals a pack value at the written precision | 0.0486 |
| 3 | number | 是 | equals a pack value at the written precision | 0.0447 |
| 4 | number | 是 | equals a pack value at the written precision | 0.0486 |
| 5 | number | 是 | equals a pack value at the written precision | 0.000081 |
| 6 | number | 是 | equals a pack value at the written precision | 0.000188 |
| 7 | cause | 是 | concept is in the attributions / linked tickets | battery: Rover_050 is at risk of a qualifying fault onset within 72 hours due to an increased battery internal resistance compared to its baseline, which has risen from 0.0447 ohm to 0.0486 ohm over the past 24 hours. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | thermal: The temperature-corrected battery internal resistance also shows a slight increase from 0.0447 ohm to 0.0486 ohm, indicating potential thermal issues. |
| 9 | cause | 是 | concept is in the attributions / linked tickets | battery: The temperature-corrected battery internal resistance also shows a slight increase from 0.0447 ohm to 0.0486 ohm, indicating potential thermal issues. |
| 10 | cause | 是 | concept is in the attributions / linked tickets | thermal: Recent operator notes mentioning 'thermal' issues have a decaying weight of 0.000081, |
| 11 | notes | 是 | linked ticket(s) exist | Recent operator notes mentioning 'thermal' issues have a decaying weight of 0.000081, |
| 12 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | vibration: 'drivetrain' issues have a decaying weight of 0.000188, suggesting these may be more relevant to the current condition. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：Attributes the temperature-corrected resistance values to the baseline-ratio feature and to '24 hours'; invents 'thermal issues'; presents the risk-lowering drivetrain-note feature as more relevant.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A10 · Rover_032 · 2026-12-08 16:00 UTC

### 1. 案例信息

- 模型风险：16.03 %（prompt 中写作 16 %）
- 告警阈值：16.03 %（prompt 中写作 16 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g4`；模型基准值（log-odds）-10.839；本条原始 margin -3.357

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.227 | 1.24 | x baseline | +1.9212 | 推高 | 是 |
| 2 | `txt_decay_battery` | recent operator notes read as 'battery' (time-decayed weight) | 6.056e-06 | 2.376e-05 | — | +1.9007 | 推高 | 是 |
| 3 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 0.1834 | 0.8496 | — | +1.5539 | 推高 | 是 |
| 4 | `txt_decay_sensor` | recent operator notes read as 'sensor' (time-decayed weight) | 5.914e-05 | 0.0005894 | — | +0.9193 | 推高 | 否 |
| 5 | `energy_ratio_to_expected_24h` | energy use relative to what terrain and ambient predict (24 h) | 0.9959 | 0.996 | x expected | +0.6604 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `vib_norm_mean_6h` | terrain-normalised vibration (6 h mean) | 0.1207 | 0.1208 | g | -0.8481 | 拉低 | 是 |
| 2 | `vib_norm_ratio_to_baseline` | terrain-normalised vibration relative to the unit's own baseline | 1.001 | 1.002 | x baseline | -0.2289 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00847 | operator_note | 117.02 | `Heavy rain today, patrol completed as planned - logging for maintenance` | no_fault_found | 0.993 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_032. Alert time: 2026-12-08 16:00 UTC.
Model risk of a qualifying fault onset within 72 h: 16 %. Alert threshold: 16 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.23 x baseline (1.24 x baseline 72 h earlier)
- recent operator notes read as 'battery' (time-decayed weight) at 0.000006 (0.000024 72 h earlier)
- recent operator notes read as 'no fault found' (time-decayed weight) at 0.183 (0.850 72 h earlier)
Features pushing the risk DOWN:
- terrain-normalised vibration (6 h mean) at 0.121 g
Operator notes for this unit in the previous 7 days (newest first):
- 117 h before the alert: "Heavy rain today, patrol completed as planned - logging for maintenance" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_032 is at high risk of a qualifying fault onset within the next 72 hours due to several factors. The battery's internal resistance has increased by 23% compared to its baseline, which is a strong indicator of potential issues. Recent operator notes mentioning 'battery' have a significant impact on the risk assessment, with a weight of 0.000006, indicating a substantial concern. Additionally, notes stating 'no fault found' have a lower but still notable weight of 0.183, suggesting ongoing operational challenges. On the other hand, the unit's recent vibration readings are within normal limits, with a mean value of 0.121 g, reducing the overall risk.
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：忠实；抽取声明 9 条，未通过 0 条。

**v2**（重新打分）：忠实；抽取声明 9 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 23 |
| 2 | number | 是 | equals a pack value at the written precision | 0.000006 |
| 3 | number | 是 | equals a pack value at the written precision | 0.183 |
| 4 | number | 是 | equals a pack value at the written precision | 0.121 |
| 5 | cause | 是 | concept is in the attributions / linked tickets | battery: The battery's internal resistance has increased by 23% compared to its baseline, which is a strong indicator of potential issues. |
| 6 | cause | 是 | concept is in the attributions / linked tickets | battery: Recent operator notes mentioning 'battery' have a significant impact on the risk assessment, with a weight of 0.000006, indicating a substantial concern. |
| 7 | notes | 是 | linked ticket(s) exist | Recent operator notes mentioning 'battery' have a significant impact on the risk assessment, with a weight of 0.000006, indicating a substantial concern. |
| 8 | notes | 是 | linked ticket(s) exist | Additionally, notes stating ' NFFCLASS ' have a lower |
| 9 | cause | 是 | concept is in the attributions / linked tickets | vibration: On the other hand, the unit's recent vibration readings are within normal limits, with a mean value of 0.121 g, reducing the overall risk. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：Claims notes 'mentioning battery' (no battery note linked; feature value 0.000006) indicate 'a substantial concern'; invents 'ongoing operational challenges'.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A11 · Rover_041 · 2026-12-09 04:00 UTC

### 1. 案例信息

- 模型风险：13.33 %（prompt 中写作 13 %）
- 告警阈值：11.84 %（prompt 中写作 12 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g0`；模型基准值（log-odds）-10.655；本条原始 margin -2.976

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `vib_norm_mean_24h` | terrain-normalised vibration (24 h mean) | 0.1424 | 0.1429 | g | +1.0629 | 推高 | 是 |
| 2 | `energy_ratio_to_expected_24h` | energy use relative to what terrain and ambient predict (24 h) | 0.9859 | 0.9858 | x expected | +0.6990 | 推高 | 是 |
| 3 | `txt_decay_thermal` | recent operator notes read as 'thermal' (time-decayed weight) | 0.0001471 | 0 | — | +0.6687 | 推高 | 是 |
| 4 | `ambient_min_24h` | 24 h minimum of ambient temperature | 17.74 | 8 | deg C | +0.5562 | 推高 | 否 |
| 5 | `hours_since_repair` | hours since the last completed repair | 1440 | 1440 | h | +0.4865 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 0.4041 | 0 | — | -0.4581 | 拉低 | 是 |
| 2 | `motor_temp_resid_dock_mean_24h` | docked motor-temperature offset from ambient (24 h mean) | 0.5256 | 0.3153 | deg C | -0.4345 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T00904 | operator_note | 62.43 | `Rover_041 - handover done, keys and charger in place. please check` | no_fault_found | 0.995 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_041. Alert time: 2026-12-09 04:00 UTC.
Model risk of a qualifying fault onset within 72 h: 13 %. Alert threshold: 12 %.
Features pushing the risk UP (strongest first):
- terrain-normalised vibration (24 h mean) at 0.142 g (0.143 g 72 h earlier)
- energy use relative to what terrain and ambient predict (24 h) at 0.986 x expected, i.e. essentially at the reference level
- recent operator notes read as 'thermal' (time-decayed weight) at 0.000147 (0 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'no fault found' (time-decayed weight) at 0.404 (0 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 62.4 h before the alert: "Rover_041 - handover done, keys and charger in place. please check" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_041 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to several factors. The most significant indicators are: - A 24-hour mean terrain-normalized vibration level of 0.142 g, which is slightly above the threshold of 0.143 g observed 72 hours prior. - An energy use rate that is 0.986 times the expected value, indicating it's operating close to its baseline performance. - Recent operator notes suggesting a thermal issue, with a weight of 0.000147, which has decreased from 0.404, suggesting a potential improvement. However, there are also mitigating factors: - Recent operator notes indicate a "no
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 11 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| notes | 1 linked operator note(s) exist | However, there are also mitigating factors: - Recent operator notes indicate a "no |

**v2**（重新打分）：不忠实；抽取声明 10 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.142 |
| 2 | number | 是 | equals a pack value at the written precision | 0.143 |
| 3 | number | 是 | equals a pack value at the written precision | 0.986 |
| 4 | number | 是 | equals a pack value at the written precision | 0.000147 |
| 5 | number | 是 | equals a pack value at the written precision | 0.404 |
| 6 | cause | 是 | concept is in the attributions / linked tickets | vibration: The most significant indicators are: - A 24-hour mean terrain-normalised vibration level of 0.142 g, which is slightly above the threshold of 0.143 g observed 72 hours prior. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | energy: - An energy use rate that is 0.986 times the expected value, indicating it's operating close to its baseline performance. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | thermal: - Recent operator notes suggesting a thermal issue, with a weight of 0.000147, which has decreased from 0.404, suggesting a potential improvement. |
| 9 | notes | 是 | linked ticket(s) exist | - Recent operator notes suggesting a thermal issue, with a weight of 0.000147, which has decreased from 0.404, suggesting a potential improvement. |
| 10 | notes | **否** | 1 linked operator note(s) exist | However, there are also mitigating factors: - Recent operator notes indicate a "no |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：false
- Claude 的说明（原文）：Calls 0.142 g 'slightly above the threshold of 0.143 g' (no such threshold; 0.143 is the 72 h value); attaches the no-fault-found value 0.404 to the thermal-note feature. Checker flag is a negation false flag.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A12 · Rover_040 · 2026-12-11 22:00 UTC

### 1. 案例信息

- 模型风险：4.91 %（prompt 中写作 4.9 %）
- 告警阈值：4.91 %（prompt 中写作 4.9 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1205_g2`；模型基准值（log-odds）-10.349；本条原始 margin -6.646

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.219 | 1.15 | x baseline | +2.4341 | 推高 | 是 |
| 2 | `txt_decay_thermal` | recent operator notes read as 'thermal' (time-decayed weight) | 6.552e-05 | 0.0001869 | — | +1.0338 | 推高 | 是 |
| 3 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 0.2712 | 0.9586 | — | +0.8163 | 推高 | 是 |
| 4 | `r_hat_temp_norm_24h` | temperature-corrected battery internal resistance (24 h) | 0.04697 | 0.0443 | ohm | +0.6357 | 推高 | 否 |
| 5 | `txt_decay_connectivity` | recent operator notes read as 'connectivity' (time-decayed weight) | 0.6397 | 0.1978 | — | +0.3005 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.0001163 | 0.0001764 | — | -0.7098 | 拉低 | 是 |
| 2 | `txt_decay_user_or_operational_error` | recent operator notes read as 'user or operational error' (time-decayed weight) | 0.0001317 | 0.0003239 | — | -0.4682 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01003 | operator_note | 30.40 | `Day shift data link unstable all morning reconnects every few mintues` | connectivity | 0.991 | 是 |
| T00901 | operator_note | 131.10 | `Morning shift: connection stable, no dropouts` | no_fault_found | 0.996 | 是 |
| T00887 | operator_note | 148.95 | `after the morning round monthly inventory check, robot present and working` | no_fault_found | 0.998 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_040. Alert time: 2026-12-11 22:00 UTC.
Model risk of a qualifying fault onset within 72 h: 4.9 %. Alert threshold: 4.9 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.22 x baseline (1.15 x baseline 72 h earlier)
- recent operator notes read as 'thermal' (time-decayed weight) at 0.000066 (0.000187 72 h earlier)
- recent operator notes read as 'no fault found' (time-decayed weight) at 0.271 (0.959 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.000116 (0.000176 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 30.4 h before the alert: "Day shift data link unstable all morning reconnects every few mintues" (ticket classifier reading: connectivity)
- 131 h before the alert: "Morning shift: connection stable, no dropouts" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_040 is at risk of a qualifying fault onset within 72 hours due to an increased battery internal resistance compared to its baseline, with a relative increase of 1.22 times. Recent operator notes indicate a thermal issue, which has a time-decayed weight of 0.000066, and a note stating "no fault found" has a weight of 0.271. The system also considers a drivetrain issue, with a weight of 0.000116, which is lower than the thermal issue. The risk is currently 4.9%, which is above the alert threshold of 4.9%.
```

句数（按存储的切分）：4；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 12 条，未通过 2 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| quote | no linked ticket has this text | no fault found |
| cause | concept is not among the features raising the risk nor in a linked ticket | vibration: The system also considers a drivetrain issue, with a weight of 0.000116, which is lower than the thermal issue. |

**v2**（重新打分）：不忠实；抽取声明 11 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 1.22 |
| 2 | number | 是 | equals a pack value at the written precision | 0.000066 |
| 3 | number | 是 | equals a pack value at the written precision | 0.271 |
| 4 | number | 是 | equals a pack value at the written precision | 0.000116 |
| 5 | number | 是 | equals a pack value at the written precision | 4.9 |
| 6 | number | 是 | equals a pack value at the written precision | 4.9 |
| 7 | cause | 是 | concept is in the attributions / linked tickets | battery: Rover_040 is at risk of a qualifying fault onset within 72 hours due to an increased battery internal resistance compared to its baseline, with a relative increase of 1.22 times. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | thermal: Recent operator notes indicate a thermal issue, which has a time-decayed weight of 0.000066, and a note stating   NFFCLASS   has a weight of 0.271. |
| 9 | notes | 是 | linked ticket(s) exist | Recent operator notes indicate a thermal issue, which has a time-decayed weight of 0.000066, and a note stating   NFFCLASS   has a weight of 0.271. |
| 10 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | vibration: The system also considers a drivetrain issue, with a weight of 0.000116, which is lower than the thermal issue. |
| 11 | cause | 是 | concept is in the attributions / linked tickets | thermal: The system also considers a drivetrain issue, with a weight of 0.000116, which is lower than the thermal issue. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：false
- Claude 的说明（原文）：Claims notes 'indicate a thermal issue' (no thermal note linked; value 0.000066). Checker flag (drivetrain mentioned) is weak; the real error is not caught.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A13 · Rover_015 · 2026-12-15 03:00 UTC

### 1. 案例信息

- 模型风险：14.31 %（prompt 中写作 14 %）
- 告警阈值：14.31 %（prompt 中写作 14 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g2`；模型基准值（log-odds）-9.783；本条原始 margin 0.247

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.6075 | 0.003564 | — | +2.1264 | 推高 | 是 |
| 2 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.538 | 1.278 | x baseline | +1.1278 | 推高 | 是 |
| 3 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0.9985 | 0.007785 | — | +1.0698 | 推高 | 是 |
| 4 | `r_bms_max_24h` | 24 h maximum of BMS-reported internal resistance | 0.08039 | 0.07355 | ohm | +0.8810 | 推高 | 否 |
| 5 | `r_bms_mean_24h` | 24 h mean of BMS-reported internal resistance | 0.07277 | 0.06561 | ohm | +0.6992 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `ambient_max_24h` | 24 h maximum of ambient temperature | 34.93 | 37.99 | deg C | -0.3133 | 拉低 | 是 |
| 2 | `motor_temp_max_24h` | 24 h maximum of motor temperature | 48.78 | 51.83 | deg C | -0.2204 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01120 | operator_note | 7.03 | `Unit 15 - voltage sag warning popped up when it accelerated. needs a look asap` | no_fault_found | 0.872 | 是 |
| T01084 | operator_note | 37.53 | `fyi plastic wrap tangled around the axle` | drivetrain | 0.998 | 是 |
| T01059 | operator_note | 56.95 | `during patrol nprmal patrol, no incidents` | no_fault_found | 0.965 | 否 |
| T01039 | operator_note | 72.03 | `MISSED HEARTBEAT ALERTS FROM THE FLEET SERVER, UNUT KEPT IN SERVICE` | connectivity | 0.960 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_015. Alert time: 2026-12-15 03:00 UTC.
Model risk of a qualifying fault onset within 72 h: 14 %. Alert threshold: 14 %.
Features pushing the risk UP (strongest first):
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.608 (0.00356 72 h earlier)
- battery internal resistance relative to the unit's own baseline at 1.54 x baseline (1.28 x baseline 72 h earlier)
- strongest fault wording in operator notes of the last 72 h (classifier probability) at 0.999 (0.00779 72 h earlier)
Features pushing the risk DOWN:
- 24 h maximum of ambient temperature at 34.9 deg C (38.0 deg C 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 7.03 h before the alert: "Unit 15 - voltage sag warning popped up when it accelerated. needs a look asap" (ticket classifier reading: no fault found)
- 37.5 h before the alert: "fyi plastic wrap tangled around the axle" (ticket classifier reading: drivetrain)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_015 is at high risk of a qualifying fault onset within the next 72 hours due to several factors. Recent operator notes indicate a drivetrain issue ("Unit 15 - voltage sag warning popped up when it accelerated. needs a look asap") with a strong likelihood of a fault based on the classifier probability of 0.999. Additionally, the battery internal resistance has increased by 25% compared to its baseline, which could be indicative of a problem. The ambient temperature has also risen slightly from 38°C to 34.9°C over the past 72 hours, but this does not significantly impact the risk assessment.
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 9 条，未通过 2 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| number | no pack value rounds to this number | 25 |
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: The ambient temperature has also risen slightly from 38°C to 34.9°C over the past 72 hours |

**v2**（重新打分）：不忠实；抽取声明 9 条，未通过 2 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | quote | 是 | matches a linked ticket | Unit 15 - voltage sag warning popped up when it accelerated. needs a look asap |
| 2 | number | 是 | equals a pack value at the written precision | 0.999 |
| 3 | number | **否** | no pack value rounds to this number | 25 |
| 4 | number | 是 | equals a pack value at the written precision | 38 |
| 5 | number | 是 | equals a pack value at the written precision | 34.9 |
| 6 | cause | 是 | concept is in the attributions / linked tickets | vibration: Recent operator notes indicate a drivetrain issue ( QUOTED ) with a strong likelihood of a fault based on the classifier probability of 0.999. |
| 7 | notes | 是 | linked ticket(s) exist | Recent operator notes indicate a drivetrain issue ( QUOTED ) with a strong likelihood of a fault based on the classifier probability of 0.999. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | battery: Additionally, the battery internal resistance has increased by 25% compared to its baseline, which could be indicative of a problem. |
| 9 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | environment: The ambient temperature has also risen slightly from 38°C to 34.9°C over the past 72 hours |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：Quotes a voltage-sag note (classified 'no fault found') as the drivetrain issue; '+25 %' for a 1.54 x baseline ratio; ambient 'risen from 38 to 34.9' (it fell).

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A14 · Rover_025 · 2026-12-16 14:00 UTC

### 1. 案例信息

- 模型风险：14.31 %（prompt 中写作 14 %）
- 告警阈值：14.31 %（prompt 中写作 14 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g2`；模型基准值（log-odds）-9.783；本条原始 margin -2.898

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.06238 | 0.1063 | — | +1.6032 | 推高 | 是 |
| 2 | `r_bms_max_24h` | 24 h maximum of BMS-reported internal resistance | 0.06331 | 0.06459 | ohm | +0.8986 | 推高 | 是 |
| 3 | `r_bms_mean_24h` | 24 h mean of BMS-reported internal resistance | 0.0549 | 0.05638 | ohm | +0.7687 | 推高 | 是 |
| 4 | `voltage_max_24h` | 24 h maximum of battery voltage | 50.65 | 50.67 | V | +0.5868 | 推高 | 否 |
| 5 | `voltage_std_24h` | 24 h variability (std) of battery voltage | 1.251 | 1.278 | V | +0.4980 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 2.682 | 1.295 | — | -1.1958 | 拉低 | 是 |
| 2 | `txt_note_count_7d` | operator notes in the last 7 days | 6 | 3 | — | -0.5459 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01150 | operator_note | 3.13 | `Note for maintenance: dock area swept` | no_fault_found | 0.972 | 是 |
| T01143 | operator_note | 21.52 | `FYI connection stable no dropouts` | no_fault_found | 0.930 | 是 |
| T01125 | operator_note | 31.42 | `AFTER THE MORNING ROUND ROUTINE CHECK COMPETE BATTERY TEMPERATURES AND WHEELS FINE` | no_fault_found | 0.966 | 否 |
| T01074 | operator_note | 75.00 | `Again toady nrml patrol, no incidents` | no_fault_found | 0.948 | 否 |
| T01010 | service_ticket | 129.73 | `chkd noise and vib, everything within limits. no falt found.` | no_fault_found | 0.987 | 否 |
| T00981 | operator_note | 162.57 | `note for maintenance thought drnoing sound from the drivetrain thhat was not there last week but it seemms fine now will monitor` | no_fault_found | 0.540 | 否 |
| T00971 | operator_note | 167.03 | `Connection stable, no dropouts, will monitor` | no_fault_found | 0.952 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_025. Alert time: 2026-12-16 14:00 UTC.
Model risk of a qualifying fault onset within 72 h: 14 %. Alert threshold: 14 %.
Features pushing the risk UP (strongest first):
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.0624 (0.106 72 h earlier)
- 24 h maximum of BMS-reported internal resistance at 0.0633 ohm (0.0646 ohm 72 h earlier)
- 24 h mean of BMS-reported internal resistance at 0.0549 ohm (0.0564 ohm 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'no fault found' (time-decayed weight) at 2.68 (1.30 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 3.13 h before the alert: "Note for maintenance: dock area swept" (ticket classifier reading: no fault found)
- 21.5 h before the alert: "FYI connection stable no dropouts" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_025 is at elevated risk of a qualifying fault onset within the next 72 hours due to several factors. Recent operator notes indicating drivetrain issues have increased the risk by 0.0624, while a high BMS-reported internal resistance of 0.0633 ohms has also pushed the risk up. The risk is further exacerbated by a 24-hour mean internal resistance of 0.0549 ohms compared to 0.0564 ohms previously. However, there are mitigating factors such as recent operator notes stating no fault was found, which slightly reduce the risk.
```

句数（按存储的切分）：4；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 9 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| notes | 6 linked operator note(s) exist | However, there are mitigating factors such as recent operator notes stating no fault was found, which slightly reduce the risk. |

**v2**（重新打分）：忠实；抽取声明 9 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 0.0624 |
| 2 | number | 是 | equals a pack value at the written precision | 0.0633 |
| 3 | number | 是 | equals a pack value at the written precision | 0.0549 |
| 4 | number | 是 | equals a pack value at the written precision | 0.0564 |
| 5 | cause | 是 | concept is in the attributions / linked tickets | vibration: Recent operator notes indicating drivetrain issues have increased the risk by 0.0624, |
| 6 | notes | 是 | linked ticket(s) exist | Recent operator notes indicating drivetrain issues have increased the risk by 0.0624, |
| 7 | cause | 是 | concept is in the attributions / linked tickets | battery: a high BMS-reported internal resistance of 0.0633 ohms has also pushed the risk up. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | battery: The risk is further exacerbated by a 24-hour mean internal resistance of 0.0549 ohms compared to 0.0564 ohms previously. |
| 9 | notes | 是 | linked ticket(s) exist | However, there are mitigating factors such as recent operator notes stating  NFFCLASS , which slightly reduce the risk. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：'Increased the risk by 0.0624': 0.0624 is the feature value, not a risk change; calls a falling resistance 'high' and 'exacerbating'.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A15 · Rover_021 · 2026-12-16 16:00 UTC

### 1. 案例信息

- 模型风险：100.00 %（prompt 中写作 100 %）
- 告警阈值：15.22 %（prompt 中写作 15 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g3`；模型基准值（log-odds）-9.065；本条原始 margin 8.041

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `vib_norm_change_72h` | change in terrain-normalised vibration over 72 h | 0.04823 | 0.02331 | g | +4.8993 | 推高 | 是 |
| 2 | `vib_norm_ratio_to_baseline` | terrain-normalised vibration relative to the unit's own baseline | 1.412 | 1.139 | x baseline | +4.2776 | 推高 | 是 |
| 3 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.015 | 1.005 | x baseline | +3.8797 | 推高 | 是 |
| 4 | `vib_norm_slope_24h` | trend of terrain-normalised vibration (24 h) | 0.00146 | 0.0003929 | g/h | +1.2729 | 推高 | 否 |
| 5 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.03911 | 0.006246 | — | +0.8834 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_battery` | recent operator notes read as 'battery' (time-decayed weight) | 0.4007 | 0.004952 | — | -0.3082 | 拉低 | 是 |
| 2 | `energy_ratio_to_expected_24h` | energy use relative to what terrain and ambient predict (24 h) | 1.022 | 1.012 | x expected | -0.3053 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01132 | operator_note | 31.72 | `unit 21: handover done, keys & charger in place` | no_fault_found | 0.972 | 是 |
| T01101 | operator_note | 54.77 | `Acting weird again, no error cde shown` | battery | 0.870 | 是 |
| T01098 | operator_note | 56.80 | `R021: normal patrol, no incidents` | no_fault_found | 0.977 | 否 |
| T01054 | operator_note | 99.02 | `Unit went offline near the warehouse corner and came back by itself` | connectivity | 0.988 | 否 |
| T01042 | operator_note | 105.25 | `service cover left open after insp needs a look asap` | user_or_operational_error | 0.968 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_021. Alert time: 2026-12-16 16:00 UTC.
Model risk of a qualifying fault onset within 72 h: 100 %. Alert threshold: 15 %.
Features pushing the risk UP (strongest first):
- change in terrain-normalised vibration over 72 h at 0.0482 g (0.0233 g 72 h earlier)
- terrain-normalised vibration relative to the unit's own baseline at 1.41 x baseline (1.14 x baseline 72 h earlier)
- terrain-normalised energy use relative to the unit's own baseline at 1.02 x baseline, i.e. essentially at the reference level (1.01 x baseline 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'battery' (time-decayed weight) at 0.401 (0.00495 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 31.7 h before the alert: "unit 21: handover done, keys & charger in place" (ticket classifier reading: no fault found)
- 54.8 h before the alert: "Acting weird again, no error cde shown" (ticket classifier reading: battery)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_021 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to significant changes in its vibration levels and energy usage compared to its baseline. The unit has shown an increase in vibration by 48% over the past 72 hours, which is above the alert threshold of 15%. Additionally, the vibration relative to the unit's baseline has risen by 27%, also exceeding the threshold. These changes suggest potential issues with the vehicle's operation or environment. However, there are mitigating factors such as recent operator notes indicating normal operation and no recent battery-related issues.
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 8 条，未通过 3 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| number | no pack value rounds to this number | 48 |
| number | no pack value rounds to this number | 27 |
| notes | 5 linked operator note(s) exist | However, there are mitigating factors such as recent operator notes indicating normal operation and no recent battery-related issues. |

**v2**（重新打分）：不忠实；抽取声明 8 条，未通过 3 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | **否** | no pack value rounds to this number | 48 |
| 2 | number | 是 | equals a pack value at the written precision | 15 |
| 3 | number | **否** | no pack value rounds to this number | 27 |
| 4 | cause | 是 | concept is in the attributions / linked tickets | vibration: Rover_021 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to significant changes in its vibration levels and energy usage compared to its baseline. |
| 5 | cause | 是 | concept is in the attributions / linked tickets | energy: Rover_021 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to significant changes in its vibration levels and energy usage compared to its baseline. |
| 6 | cause | 是 | concept is in the attributions / linked tickets | vibration: The unit has shown an increase in vibration by 48% over the past 72 hours, which is above the alert threshold of 15%. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | vibration: Additionally, the vibration relative to the unit's baseline has risen by 27%, also exceeding the threshold. |
| 8 | notes | **否** | 5 linked operator note(s) exist | However, there are mitigating factors such as recent operator notes indicating normal operation and no recent battery-related issues. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：Invents '+48 %', '+27 %' and an 'alert threshold of 15 %' for vibration (15 % is the risk threshold).

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A16 · Rover_012 · 2026-12-18 12:00 UTC

### 1. 案例信息

- 模型风险：55.32 %（prompt 中写作 55 %）
- 告警阈值：55.32 %（prompt 中写作 55 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g1`；模型基准值（log-odds）-10.250；本条原始 margin -0.029

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.24 | 1.072 | x baseline | +2.6247 | 推高 | 是 |
| 2 | `r_bms_max_24h` | 24 h maximum of BMS-reported internal resistance | 0.07401 | 0.06981 | ohm | +2.4767 | 推高 | 是 |
| 3 | `r_bms_min_24h` | 24 h minimum of BMS-reported internal resistance | 0.05294 | 0.05328 | ohm | +1.1683 | 推高 | 是 |
| 4 | `r_bms_mean_24h` | 24 h mean of BMS-reported internal resistance | 0.06291 | 0.06148 | ohm | +0.9768 | 推高 | 否 |
| 5 | `motor_temp_max_24h` | 24 h maximum of motor temperature | 34.83 | 28.5 | deg C | +0.7957 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `thermal_gain_ratio_24h` | motor heating per watt relative to the nominal model (24 h) | 0.9233 | 0.9368 | x nominal | -0.8755 | 拉低 | 是 |
| 2 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | -0.9065 | -0.7808 | deg C | -0.5225 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01180 | operator_note | 21.90 | `Rover 12: no alarms during the night shift. logging for maintenance` | no_fault_found | 0.996 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_012. Alert time: 2026-12-18 12:00 UTC.
Model risk of a qualifying fault onset within 72 h: 55 %. Alert threshold: 55 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.24 x baseline (1.07 x baseline 72 h earlier)
- 24 h maximum of BMS-reported internal resistance at 0.0740 ohm (0.0698 ohm 72 h earlier)
- 24 h minimum of BMS-reported internal resistance at 0.0529 ohm (0.0533 ohm 72 h earlier)
Features pushing the risk DOWN:
- motor heating per watt relative to the nominal model (24 h) at 0.923 x nominal (0.937 x nominal 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 21.9 h before the alert: "Rover 12: no alarms during the night shift. logging for maintenance" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_012 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to several factors. The battery internal resistance has increased by 24% compared to its baseline value, which is a strong indicator of potential issues. Additionally, the BMS-reported internal resistance has shown a significant drop from 0.0698 ohm to 0.0529 ohm over the past 24 hours, suggesting a decrease in resistance. These changes push the risk up to 55%, which is above the alert threshold of 55%. However, there are also mitigating factors such as lower motor heating rates, which could potentially reduce the overall risk. The operator should closely monitor the unit's performance and log any further changes
```

句数（按存储的切分）：6；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：忠实；抽取声明 8 条，未通过 0 条。

**v2**（重新打分）：忠实；抽取声明 8 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 24 |
| 2 | number | 是 | equals a pack value at the written precision | 0.0698 |
| 3 | number | 是 | equals a pack value at the written precision | 0.0529 |
| 4 | number | 是 | equals a pack value at the written precision | 55 |
| 5 | number | 是 | equals a pack value at the written precision | 55 |
| 6 | cause | 是 | concept is in the attributions / linked tickets | battery: The battery internal resistance has increased by 24% compared to its baseline value, which is a strong indicator of potential issues. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | battery: Additionally, the BMS-reported internal resistance has shown a significant drop from 0.0698 ohm to 0.0529 ohm over the past 24 hours, suggesting a decrease in resistance. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | thermal: However, there are also mitigating factors such as lower motor heating rates, which could potentially reduce the overall risk. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：Mixes the 72 h-earlier maximum (0.0698) with the current minimum (0.0529) into an invented 'significant drop'; adds advice ('the operator should closely monitor'), which the prompt forbids.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A17 · Rover_049 · 2026-12-18 23:00 UTC

### 1. 案例信息

- 模型风险：16.67 %（prompt 中写作 17 %）
- 告警阈值：16.67 %（prompt 中写作 17 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g4`；模型基准值（log-odds）-9.209；本条原始 margin -2.999

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.278 | 1.297 | x baseline | +3.8501 | 推高 | 是 |
| 2 | `thermal_gain_ratio_24h` | motor heating per watt relative to the nominal model (24 h) | 0.9758 | 0.9799 | x nominal | +1.3405 | 推高 | 是 |
| 3 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0.973 | 0.01738 | — | +0.7765 | 推高 | 是 |
| 4 | `txt_decay_battery` | recent operator notes read as 'battery' (time-decayed weight) | 0.8887 | 0.009779 | — | +0.6459 | 推高 | 否 |
| 5 | `motor_temp_resid_patrol_mean_24h` | motor temperature above the load/ambient model while active (24 h mean) | -0.2844 | -0.2363 | deg C | +0.5041 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `terrain_patrol_mean_24h` | mean terrain load while active (24 h) | 0.4695 | 0.4675 | — | -0.5670 | 拉低 | 是 |
| 2 | `ambient_mean_24h` | 24 h mean of ambient temperature | 31.61 | 30.11 | deg C | -0.4718 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01208 | operator_note | 6.53 | `charge contacts dirty, cleaned them & it charges again, needs a look asap` | battery | 0.967 | 是 |
| T01196 | operator_note | 10.43 | `Connection stable, no dropouts, reported by the guard on site` | no_fault_found | 0.895 | 是 |
| T01122 | operator_note | 93.53 | `Rover_049 no data uplink during patr of the patrol same as lst shift` | connectivity | 0.845 | 否 |
| T01112 | operator_note | 104.45 | `Rover_049: temps normal all day, second time this week` | no_fault_found | 0.968 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_049. Alert time: 2026-12-18 23:00 UTC.
Model risk of a qualifying fault onset within 72 h: 17 %. Alert threshold: 17 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.28 x baseline (1.30 x baseline 72 h earlier)
- motor heating per watt relative to the nominal model (24 h) at 0.976 x nominal (0.980 x nominal 72 h earlier)
- strongest fault wording in operator notes of the last 72 h (classifier probability) at 0.973 (0.0174 72 h earlier)
Features pushing the risk DOWN:
- mean terrain load while active (24 h) at 0.469 (0.467 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 6.53 h before the alert: "charge contacts dirty, cleaned them & it charges again, needs a look asap" (ticket classifier reading: battery)
- 10.4 h before the alert: "Connection stable, no dropouts, reported by the guard on site" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_049 is experiencing a high risk of a qualifying fault onset within the next 72 hours due to several factors. The battery internal resistance has increased by 28% compared to its baseline value, indicating potential degradation. Additionally, the motor heating has risen by 9.76%, which could be indicative of overheating issues. The most recent operator note mentions a faulty charge contact that was cleaned, but the ticket classifier reading indicates that the issue was resolved. However, the classifier probability for the strongest fault wording in the notes remains high at 97.3%. Despite some positive indicators, such as stable connections and no dropouts, the overall risk is elevated due to these factors.
```

句数（按存储的切分）：6；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 8 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| number | no pack value rounds to this number | 9.76 |

**v2**（重新打分）：不忠实；抽取声明 8 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 28 |
| 2 | number | **否** | no pack value rounds to this number | 9.76 |
| 3 | number | 是 | equals a pack value at the written precision | 97.3 |
| 4 | cause | 是 | concept is in the attributions / linked tickets | battery: The battery internal resistance has increased by 28% compared to its baseline value, indicating potential degradation. |
| 5 | cause | 是 | concept is in the attributions / linked tickets | thermal: Additionally, the motor heating has risen by 9.76%, which could be indicative of overheating issues. |
| 6 | notes | 是 | linked ticket(s) exist | The most recent operator note mentions a faulty charge contact that was cleaned |
| 7 | notes | 是 | linked ticket(s) exist | the ticket classifier reading indicates that the issue was resolved. |
| 8 | notes | 是 | linked ticket(s) exist | However, the classifier probability for the strongest fault wording in the notes remains high at 97.3%. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：'Motor heating has risen by 9.76 %' (it is 0.976 x nominal, below nominal); invents 'overheating issues' and that the classifier reading says the issue was resolved.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A18 · Rover_011 · 2026-12-20 05:00 UTC

### 1. 案例信息

- 模型风险：14.31 %（prompt 中写作 14 %）
- 告警阈值：14.31 %（prompt 中写作 14 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g2`；模型基准值（log-odds）-9.783；本条原始 margin -4.332

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.317 | 1.091 | x baseline | +2.0561 | 推高 | 是 |
| 2 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0.9457 | 0.02344 | — | +0.7508 | 推高 | 是 |
| 3 | `motor_temp_max_24h` | 24 h maximum of motor temperature | 41.47 | 41.07 | deg C | +0.5369 | 推高 | 是 |
| 4 | `r_bms_mean_24h` | 24 h mean of BMS-reported internal resistance | 0.05273 | 0.04772 | ohm | +0.5344 | 推高 | 否 |
| 5 | `r_bms_min_24h` | 24 h minimum of BMS-reported internal resistance | 0.04675 | 0.04181 | ohm | +0.5280 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.007773 | 0.01002 | — | -0.6226 | 拉低 | 是 |
| 2 | `txt_decay_no_fault_found` | recent operator notes read as 'no fault found' (time-decayed weight) | 1.307 | 0.8266 | — | -0.3704 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01206 | operator_note | 37.30 | `Note for maint did not finish the full patrol loop low batt warning at 15%` | battery | 0.940 | 是 |
| T01193 | operator_note | 42.00 | `Again today charging normally and ran the full shift` | no_fault_found | 0.921 | 是 |
| T01189 | operator_note | 46.05 | `temps normal all day` | no_fault_found | 0.976 | 否 |
| T01163 | operator_note | 82.50 | `qyiet shift` | no_fault_found | 0.961 | 否 |
| T01087 | operator_note | 158.35 | `r011 - pre-shift check was skipped & the patrol started late` | user_or_operational_error | 0.986 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_011. Alert time: 2026-12-20 05:00 UTC.
Model risk of a qualifying fault onset within 72 h: 14 %. Alert threshold: 14 %.
Features pushing the risk UP (strongest first):
- battery internal resistance relative to the unit's own baseline at 1.32 x baseline (1.09 x baseline 72 h earlier)
- strongest fault wording in operator notes of the last 72 h (classifier probability) at 0.946 (0.0234 72 h earlier)
- 24 h maximum of motor temperature at 41.5 deg C (41.1 deg C 72 h earlier)
Features pushing the risk DOWN:
- recent operator notes read as 'drivetrain' (time-decayed weight) at 0.00777 (0.0100 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 37.3 h before the alert: "Note for maint did not finish the full patrol loop low batt warning at 15%" (ticket classifier reading: battery)
- 42.0 h before the alert: "Again today charging normally and ran the full shift" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_011 is at elevated risk of a qualifying fault onset within the next 72 hours due to several factors. The battery's internal resistance has increased by 32% compared to its baseline, which is a strong indicator of potential degradation. Additionally, the most recent operator note mentioning a drivetrain issue has a high classifier probability of 94.6%, suggesting a significant concern. Furthermore, the motor temperature reached 41.5 degrees Celsius, which is 0.4 degrees higher than 72 hours ago. These conditions, combined with the recent absence of any fault-related issues, indicate a heightened risk of a fault occurring.
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 8 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | vibration: Additionally, the most recent operator note mentioning a drivetrain issue has a high classifier probability of 94.6%, suggesting a significant concern. |

**v2**（重新打分）：不忠实；抽取声明 8 条，未通过 1 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 32 |
| 2 | number | 是 | equals a pack value at the written precision | 94.6 |
| 3 | number | 是 | equals a pack value at the written precision | 41.5 |
| 4 | number | 是 | equals a pack value at the written precision | 0.4 |
| 5 | cause | 是 | concept is in the attributions / linked tickets | battery: The battery's internal resistance has increased by 32% compared to its baseline, which is a strong indicator of potential degradation. |
| 6 | cause | **否** | concept is not among the features raising the risk nor in a linked ticket | vibration: Additionally, the most recent operator note mentioning a drivetrain issue has a high classifier probability of 94.6%, suggesting a significant concern. |
| 7 | notes | 是 | linked ticket(s) exist | Additionally, the most recent operator note mentioning a drivetrain issue has a high classifier probability of 94.6%, suggesting a significant concern. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | thermal: Furthermore, the motor temperature reached 41.5 degrees Celsius, which is 0.4 degrees higher than 72 hours ago. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：true
- Claude 的说明（原文）：Says the strongest-fault note mentions a drivetrain issue (it is a battery note; the drivetrain-note feature lowers the risk).

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A19 · Rover_036 · 2026-12-20 15:00 UTC

### 1. 案例信息

- 模型风险：14.31 %（prompt 中写作 14 %）
- 告警阈值：14.31 %（prompt 中写作 14 %）
- 实际结果：`false_alarm`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g2`；模型基准值（log-odds）-9.783；本条原始 margin -4.340

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.001 | 1.001 | x baseline | +1.5997 | 推高 | 是 |
| 2 | `vibration_max_24h` | 24 h maximum of vibration | 0.239 | 0.2498 | g | +0.6537 | 推高 | 是 |
| 3 | `r_bms_mean_24h` | 24 h mean of BMS-reported internal resistance | 0.05001 | 0.04898 | ohm | +0.5212 | 推高 | 是 |
| 4 | `bms_soh_last` | latest BMS state-of-health estimate | 97.55 | 97.21 | % | +0.5062 | 推高 | 否 |
| 5 | `motor_temp_resid_dock_mean_24h` | docked motor-temperature offset from ambient (24 h mean) | 0.2725 | 0.2142 | deg C | +0.4251 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `txt_fault_max_72h` | strongest fault wording in operator notes of the last 72 h (classifier probability) | 0 | 0.05794 | — | -0.4717 | 拉低 | 是 |
| 2 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.007818 | 0.02211 | — | -0.4335 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01148 | operator_note | 102.33 | `Handover done, keys & charger in place` | no_fault_found | 0.984 | 是 |
| T01134 | operator_note | 125.57 | `Robot 36 IMU needs recalibration heading drifts while stationary` | sensor | 0.802 | 是 |
| T01093 | operator_note | 159.47 | `note for maiint: lte signal weak, commands delayed` | connectivity | 0.934 | 否 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_036. Alert time: 2026-12-20 15:00 UTC.
Model risk of a qualifying fault onset within 72 h: 14 %. Alert threshold: 14 %.
Features pushing the risk UP (strongest first):
- terrain-normalised energy use relative to the unit's own baseline at 1.00 x baseline, i.e. essentially at the reference level
- 24 h maximum of vibration at 0.239 g (0.250 g 72 h earlier)
- 24 h mean of BMS-reported internal resistance at 0.0500 ohm (0.0490 ohm 72 h earlier)
Features pushing the risk DOWN:
- strongest fault wording in operator notes of the last 72 h (classifier probability) at 0 (0.0579 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 102 h before the alert: "Handover done, keys & charger in place" (ticket classifier reading: no fault found)
- 126 h before the alert: "Robot 36 IMU needs recalibration heading drifts while stationary" (ticket classifier reading: sensor)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_036 is currently at a high risk of experiencing a qualifying fault within the next 72 hours due to several factors. The unit has shown signs of increased energy usage compared to its baseline, with a 1.00x increase in terrain-normalized energy use. Additionally, there have been two instances of high vibration levels, reaching 0.239g, which is 0.001g higher than the previous 72-hour period. Furthermore, the unit's internal resistance has dropped from 0.0490 ohms to 0.0500 ohms over the past 72 hours, indicating potential issues with battery health. Despite these indicators, the risk has been pushed up by the presence of strong fault words
```

句数（按存储的切分）：5；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 9 条，未通过 1 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: The unit has shown signs of increased energy usage compared to its baseline, with a 1.00x increase in terrain-normalized energy use. |

**v2**（重新打分）：忠实；抽取声明 8 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 1.00 |
| 2 | number | 是 | equals a pack value at the written precision | 0.239 |
| 3 | number | 是 | equals a pack value at the written precision | 0.001 |
| 4 | number | 是 | equals a pack value at the written precision | 0.0490 |
| 5 | number | 是 | equals a pack value at the written precision | 0.0500 |
| 6 | cause | 是 | concept is in the attributions / linked tickets | energy: The unit has shown signs of increased energy usage compared to its baseline, with a 1.00x increase in terrain-normalised energy use. |
| 7 | cause | 是 | concept is in the attributions / linked tickets | vibration: Additionally, there have been two instances of high vibration levels, reaching 0.239g, which is 0.001g higher than the previous 72-hour period. |
| 8 | cause | 是 | concept is in the attributions / linked tickets | battery: Furthermore, the unit's internal resistance has dropped from 0.0490 ohms to 0.0500 ohms over the past 72 hours, indicating potential issues with battery health. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：Invents 'two instances' of high vibration and says 0.239 g is higher than 0.250 g; says resistance 'dropped' from 0.0490 to 0.0500; presents the risk-lowering fault-wording feature as pushing the risk up.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：

---

## A20 · Rover_016 · 2026-12-21 10:00 UTC

### 1. 案例信息

- 模型风险：4.74 %（prompt 中写作 4.7 %）
- 告警阈值：4.74 %（prompt 中写作 4.7 %）
- 实际结果：`true_positive`（真阳性 / 误报；仅供参考，忠实度只看说明与证据是否一致）
- 折：`c1215_g0`；模型基准值（log-odds）-9.631；本条原始 margin -4.691

### 2. 证据包（完整）与 LLM 收到的 prompt

**推高风险的特征（SHAP > 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `vib_norm_slope_24h` | trend of terrain-normalised vibration (24 h) | 0.001165 | 无有效读数（缺失） | g/h | +2.5356 | 推高 | 是 |
| 2 | `energy_ratio_to_baseline` | terrain-normalised energy use relative to the unit's own baseline | 1.001 | 1 | x baseline | +2.0167 | 推高 | 是 |
| 3 | `vib_norm_ratio_to_baseline` | terrain-normalised vibration relative to the unit's own baseline | 1.057 | 0.9987 | x baseline | +1.5988 | 推高 | 是 |
| 4 | `r_hat_ratio_to_baseline` | battery internal resistance relative to the unit's own baseline | 1.141 | 1.149 | x baseline | +0.7039 | 推高 | 否 |
| 5 | `motor_temp_resid_dock_mean_24h` | docked motor-temperature offset from ambient (24 h mean) | 0.1856 | 0.1147 | deg C | +0.5742 | 推高 | 否 |

**拉低风险的特征（SHAP < 0，按贡献从大到小）**

| # | 特征代码 | 特征描述 | 当前值 | 72 h 前 | 单位 | SHAP | 方向 | 进入 prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | `energy_ratio_to_expected_24h` | energy use relative to what terrain and ambient predict (24 h) | 0.9896 | 0.9882 | x expected | -0.4398 | 拉低 | 是 |
| 2 | `txt_decay_drivetrain` | recent operator notes read as 'drivetrain' (time-decayed weight) | 0.001559 | 0.0001413 | — | -0.3786 | 拉低 | 否 |

**告警前 7 天的关联记录**（文本为合成数据，原样照录；分类器判读不是真实标签）

| 编号 | 类型 | 距告警 (h) | 原文 | 分类器判读 | 概率 | 进入 prompt |
|---|---|---|---|---|---|---|
| T01245 | service_ticket | 15.95 | `Bench test passed, all readings in spec. No fault found, RTS.` | no_fault_found | 0.994 | 否 |
| T01210 | operator_note | 64.75 | `Customer reports pulls slightly to the front left on straight sections, not confirmed, reported by the guard on site` | no_fault_found | 0.953 | 是 |
| T01207 | operator_note | 65.57 | `Visitors asked about the robot, no issues` | no_fault_found | 0.999 | 是 |

**LLM 收到的 user prompt（原样）**

```text
EVIDENCE
Unit: Rover_016. Alert time: 2026-12-21 10:00 UTC.
Model risk of a qualifying fault onset within 72 h: 4.7 %. Alert threshold: 4.7 %.
Features pushing the risk UP (strongest first):
- trend of terrain-normalised vibration (24 h) at 0.00116 g/h
- terrain-normalised energy use relative to the unit's own baseline at 1.00 x baseline, i.e. essentially at the reference level
- terrain-normalised vibration relative to the unit's own baseline at 1.06 x baseline (0.999 x baseline 72 h earlier)
Features pushing the risk DOWN:
- energy use relative to what terrain and ambient predict (24 h) at 0.990 x expected, i.e. essentially at the reference level (0.988 x expected 72 h earlier)
Operator notes for this unit in the previous 7 days (newest first):
- 64.8 h before the alert: "Customer reports pulls slightly to the front left on straight sections, not confirmed, reported by the guard on site" (ticket classifier reading: no fault found)
- 65.6 h before the alert: "Visitors asked about the robot, no issues" (ticket classifier reading: no fault found)
Write the explanation now.
```

### 3. LLM 原始输出（原样，未修改、未截断、未意译）

```text
Rover_016 is currently experiencing a high risk of a qualifying fault onset within the next 72 hours, with a risk of 4.7%. The risk is being pushed up by three main factors: a 24-hour trend of terrain-normalized vibration at 0.00116 g/h, indicating potential mechanical wear; a 24-hour terrain-normalized energy use relative to the unit's baseline at 1.00 times the reference level, suggesting inefficient operation; and a 24-hour terrain-normalized vibration relative to the unit's baseline at 1.06 times the reference level, which is 0.999 times its value 72 hours ago. These factors indicate potential mechanical issues that could lead to a fault
```

句数（按存储的切分）：3；系统指令要求 2–3 句。

### 4. 自动检查器结论

**v1**（原始实验）：不忠实；抽取声明 11 条，未通过 3 条。

| 类型 | 检查器给出的原因 | 被检查的文本 |
|---|---|---|
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: The risk is being pushed up by three main factors: a 24-hour trend of terrain-normalized vibration at 0.00116 g/h, indicating potential mechanical wear |
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: a 24-hour terrain-normalized energy use relative to the unit's baseline at 1.00 times the reference level, suggesting inefficient operation |
| cause | concept is not among the features raising the risk nor in a linked ticket | environment: and a 24-hour terrain-normalized vibration relative to the unit's baseline at 1.06 times the reference level, which is 0.999 times its value 72 hours ago. |

**v2**（重新打分）：忠实；抽取声明 8 条，未通过 0 条。全部声明：

| # | 类型 | 通过 | 检查器给出的原因 | 被检查的文本（检查器内部预处理后，特征名可能被替换为 FEATURE 等标记） |
|---|---|---|---|---|
| 1 | number | 是 | equals a pack value at the written precision | 4.7 |
| 2 | number | 是 | equals a pack value at the written precision | 0.00116 |
| 3 | number | 是 | equals a pack value at the written precision | 1.00 |
| 4 | number | 是 | equals a pack value at the written precision | 1.06 |
| 5 | number | 是 | equals a pack value at the written precision | 0.999 |
| 6 | cause | 是 | concept is in the attributions / linked tickets | vibration: The risk is being pushed up by three main factors: a 24-hour trend of terrain-normalised vibration at 0.00116 g/h, indicating potential mechanical wear |
| 7 | cause | 是 | concept is in the attributions / linked tickets | energy: a 24-hour terrain-normalised energy use relative to the unit's baseline at 1.00 times the reference level, suggesting inefficient operation |
| 8 | cause | 是 | concept is in the attributions / linked tickets | vibration: and a 24-hour terrain-normalised vibration relative to the unit's baseline at 1.06 times the reference level, which is 0.999 times its value 72 hours ago. |

### 5. 已有实验分析（Claude Opus 5.5 此前的判读，仅供参考，不是审核结论）

- Claude 判读：不忠实
- Claude 对检查器标记是否有效的看法（auto_flag_valid）：null（检查器未标记）
- Claude 的说明（原文）：Invents 'inefficient operation' for 1.00 x baseline energy and 'mechanical wear'; garbles the 72 h comparison.

### 审核判定（待审核者填写）

- faithful：
- 主要原因：
- 次要问题：
- 格式问题（单独记录，不作为主要判定依据）：
- auto_flag_valid（检查器 v2 若有标记，标记是否指向真实错误；未标记则为 null）：
