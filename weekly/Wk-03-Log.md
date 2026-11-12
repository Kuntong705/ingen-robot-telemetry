# Week 03 — Engineering Log

**Project:** Robot Fleet Telemetry Analytics & Predictive Health  
**Period:** November 9 – 14, 2026 (Week 3 scope; written November 12; all data synthetic)  
**Topic:** How maintenance text enters the early-warning model, and what did not carry over from the news-sentiment pipeline

## What transferred directly

The shape of the stock project carried over almost unchanged. Text is turned into a per-document score by a classifier (TF-IDF, then sentence embeddings). The scores are aggregated as-of onto the prediction grid with a time decay. A fusion layer combines them with the numeric signal, and an LLM layer explains the result. The leakage reflexes also carried over: grouped splits, an embargo, and nothing from after the prediction time.

## The decision: what the text feature is allowed to know

In the news pipeline the sentiment of an article is known the moment it is published. A maintenance note is different. Its category is confirmed only when a technician triages it or closes the repair, often days later. I considered three designs:

1. **Train the ticket classifier once on the whole corpus** and use its probabilities as features, the natural choice for news sentiment. It is simple, but every training row would carry an in-sample score, and test-unit labels would leak into the classifier.
2. **Feed raw note embeddings straight into the early-warning model.** This needs no labels, but closure notes ("replaced bearing") would then encode the outcome, and nothing stops a repair note from predicting its own fault.
3. **Fit the classifier inside each fold** on training-unit tickets whose label was *confirmed* before the cut-off, cross-fit it over unit groups, and aggregate only probabilities of notes created before *t*. I chose this one.

The cost of option 3 is fewer training labels per fold (about 550 tickets on average, 280 at the first cut-off), and the classifier is weakest on early symptom notes (33 % accuracy). It is the only option whose result I could defend.

## What turned out to be different

- **Sparsity.** News arrives every day. Only 9 of 31 test-period fault onsets had a symptom note in the 72 h before them. Early fusion changed PR-AUC by +0.016 [−0.040, 0.078] over telemetry LightGBM. Stacking text onto the Isolation Forest cost −0.153, because the stacker over-weighted a noisy modality fitted on a few validation events. At 4× the assumed notice rate, text adds +0.130 [0.044, 0.230]. So the honest answer depends on how often real operators write, which InGen's ticket history can measure.
- **"No fault found" dominates.** It is 61 % of the test tickets, and real symptom notes read like it.
- **LLM faithfulness.** The stock project used the LLM to make models more interpretable; here every summary had to be checked against the attributions. The template summaries passed 20/20, both automatically and by hand. The local 1.5 B model passed 0/20 by hand, while a lexical checker passed 8 of them. Plausible-sounding is not faithful.

## Next

Ask InGen for the share of faults with a pre-onset note. Ship the template summaries with the Week 4 dashboard.
