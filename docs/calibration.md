# Calibration

A raw probability depends on the content **and** on how the question and its answers are worded. Every answer comes
with extra numbers so the scores can be made comparable.

## The numbers

- **`p`**: the probability of an answer among the answers you listed (they sum to 1), read from the model's next-word
  distribution at the end of your template.
- **`logprob`**: the model's log probability of the answer itself, before that normalisation: the exact string (its
  tokens' log probabilities summed), with its case variants (`rise`, `Rise`, `RISE`) added when `case_insensitive` is on.
- **`floor_p`** and **`floor_residual`**: the same, scored with **empty** content. This is the question's built-in bias.
  The lift `p − floor_p` is what the content itself says. The floor of a question is computed once per call, however
  many contents share it.
- **`residual`** (per question and combination): `1 − Σ P(answers)`, the probability that went to anything other than
  your answers. See below.

## Reading the residual

The residual is mostly set by the **template and the words**, not by the content:

- After a template like `"…\nAnswer:{?}"` the model also expects many other words, so even a good one-word question can
  have a residual of 0.8 to 0.9. "Will oil prices rise or fall next month?" on a news headline gave `p(" rise") = 0.64`
  against a floor of 0.56, with a residual of 0.92.
- For answers of **several tokens** (`" energy sector"`), the probability of the exact string is small, and the
  residual is close to 1 however well the answer fits. `p` still separates the answers: 0.98 for `" energy sector"` on
  an Exxon headline.

So judge a question on **`p` and the lift**, not on the residual's size. The residual is useful **within one
question**: a content whose residual is much higher than usual for that question is one the question does not fit
well, and deserves less weight. The engine adds a `low_option_mass` warning when a question's one-word answers take
less than 10% of the probability; take it as a hint to check the words and the template.

## Choosing a template

The template is part of the question. Rewording it changes what the model is asked, and so `p` and the lift, not only
the residual. Naming the answers in the template ("… Answer rise or fall.") lowers the residual, but in our tests it
also moved some lifts from positive to negative. Before relying on a template:

- score a few contents you know the answer to, with two or three wordings, and keep the one that separates them best;
- then keep it fixed: scores are comparable over time only under the same template, answers and `system_fingerprint`.

## Calibrating against a baseline

A **cell** is one score: one question (a template with one combination of its placeholders) on one piece of content.
A raw cell carries the bias of its wording, so calibrate it before you compare or aggregate it:

1. Score a baseline set of documents of the same kind: for example a random sample of articles from the same sources
   and period.
2. For each question and answer, keep the distribution of its lift over the baseline.
3. Express each new score as a percentile or z-score of that distribution. This gives an absolute, comparable
   probability that is stable across questions, sources and time.

## Macro cells: many cells, one better-calibrated read

Don't rely on one cell for something you want to measure. Ask it through several cells, and combine them:

- **Several wordings** of the same question (3 to 5), each calibrated on its own.
- **Related questions** that point the same way, such as "Does the article report a supply cut?" and "Is supply
  getting tighter?" (flip the sign of those that point the other way).
- **Related placeholder values**, such as `{commodity}` = crude oil, Brent, WTI.

Average their calibrated scores into a **macro cell**. The wording bias and the noise of each cell partly cancel, so
the macro cell is better calibrated and more stable than any single cell. The extra cells cost little: the content is
read once per call, and each question adds only its own tokens.

Then aggregate the macro cells over many documents (for example per day, per company or per segment), and standardise
each aggregate on its own trailing window. The prompt packs are built this way: many reads per category, each
standardised on normal traffic, averaged, then mapped to a 0 to 1 score ([Prompt packs](prompt-packs.md)).

For decisions on single items, calibrate on labelled items instead: choose the threshold on `p`, or on a macro cell's
score, that gives the precision and recall you need.

The floor depends only on the question, so for a fixed set of questions you can keep the floors of a first call and
send `"empty_floor": false` afterwards; recompute them when the `system_fingerprint` changes. On the fp8s scheme a
score does not depend on what else is in the call, so floors from different calls are identical.

## Extraction probabilities

An extraction slot returns the most likely answers with a probability each ([Scoring API](scoring-api.md#extraction-and-generation-slots)).
That `p` is a different object from a scored question's: there is no fixed list of answers and no empty floor, so it is
calibrated with labels rather than against a baseline. What the maintainers measured, accuracy of the top answer:

| task | 12B | 31B |
|---|---|---|
| the brand of a TV commercial, 32 frames | 0.97 | 0.99 |
| the same from 4 frames / 1 frame | 0.57 / 0.19 | 0.74 / 0.29 |
| a dog's breed from a photo (Imagewoof) | 0.51 | 0.65 |
| a span from a paragraph (SQuAD, exact match; token F1 0.69 / 0.70) | 0.59 | 0.61 |
| a fact from memory, no content (Natural Questions) | 0.20 | 0.28 |
| a number from a paragraph (SQuAD numeric, `{n}`) | 0.95 | |
| a number from a financial report (TAT-QA, `{$}`) | 0.90 | |

- **The probability ranks well.** The raw `p` separates right from wrong answers: AUC 0.72 to 0.92 on the 12B, 0.68 to
  0.90 on the 31B, on every task.
- **The raw `p` is not P(correct): it is under-confident**, by an amount that depends on the task. It averages about
  half the accuracy (SQuAD: a mean `p` of 0.19 for 59 % right): a right answer's probability is split over forms the
  merging cannot join ("Coke" and "Coca-Cola", spans a word longer or shorter). Numbers are less under-confident (a raw
  calibration error of about 0.09 on some sets), but still under on easy ones.
- **Calibrate per field with about 50 labelled examples.** Fit `P(correct) = sigmoid(a·logit(p) + b)` on
  (logit p, right or wrong) pairs with a logistic fit in any statistics library (spinf-client has a helper), send the
  two numbers as the query's `calibration: {a, b}`, and read `p_calibrated` on the top answer. With 50 labels the
  calibration error was 0.02 to 0.10 on every task and both models.
- **Without labels**, a pooled map is right on average but can be off by about 0.2 on one field: 12B
  `sigmoid(0.76·logit(p) + 1.42)`, 31B `sigmoid(0.71·logit(p) + 1.34)`; the correction depends on the task, hardly on
  the model. A map fitted on answers found in the content over-corrects answers from memory.
- **Selective answering works.** Keep the answers above a calibrated threshold: for numbers with the pooled map, P ≥ 0.8
  keeps 75 % of the answers at 90 % right, P ≥ 0.9 keeps 39 % at 96 %; for text on the 12B, P ≥ 0.8 keeps 14 % at 88 %.
- **Tried, not useful:** asking the model whether its answer is right (AUC 0.52 to 0.69: a base model says yes to almost
  anything); the gap to the second answer, `mass` or the number of answers as confidence features; renormalising over
  the answers found (over-confident when the model guesses).
- **The 31B is clearly better on hard inputs** (few frames, fine-grained images, memory); on easy ones both are near the
  ceiling.
