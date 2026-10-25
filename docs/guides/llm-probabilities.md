# Probabilities from a language model: confidence scores you can use

A language model asked to classify something usually answers with text: a label, a sentence, sometimes a paragraph.
Code cannot branch on a paragraph, and a label on its own does not say how sure the model was. What most
classification jobs need is a number: the probability of each possible answer, which you can threshold, rank, average
and compare over time.

This guide explains how spinf-serve returns that number, what it means, and how to turn it into a calibrated score.
The same request runs unchanged against the hosted API at api.spinf.com with a key; everything below is about the
engine you run yourself.

## A probability over the answers you allow

With spinf-serve you do not ask an open question and parse the reply. You write the question as a template that ends
where the answer goes, and you list the answers you accept:

```json
{
  "id": "supply_cut",
  "template": "\n\nDoes the article report a cut in oil supply?\nAnswer:{?}",
  "options": [" yes", " no"]
}
```

For each answer, the engine returns `p`: the probability the model gives that answer at the end of your template,
normalised over the answers you listed, so they sum to 1. Nothing is generated for a scored question. The answer is
always one of yours, with its probability, so there is no free text to parse and no invented label to handle. (For an
open field such as a name or an amount, an extraction slot lets the model write the answer and still returns
probabilities: [Scoring API](../scoring-api.md#extraction-and-generation-slots).)

The same works for any closed question: a yes/no probability, a topic among five, a sentiment among three, a queue
among twenty. It is classification with confidence built in, rather than a label with a confidence you have to guess.

## Why not ask the model for its confidence?

A common workaround is to ask the model to write its own score ("rate your confidence from 0 to 100"). The number it
writes is more text: it tends to cluster on a few round values, changes with the wording, and is not tied to the
probabilities the model actually assigns. A probability read over a fixed set of answers is a measurement of the
model's own prediction, on a scale that stays the same for every item.

## The question has a bias: the empty-content floor

Every question leans towards some answers before it reads anything. A template ending in "Answer:" after a yes/no
question may favour " yes" whatever the content. To measure this, spinf-serve also scores each question with empty
content and returns the result as `floor_p`.

In the [quickstart](../quickstart.md) call, a news article about OPEC+ extending output cuts gives, on the 12B
(`spinf/gemma-4-12b-fp8`):

| Answer | `p` | `floor_p` |
|---|---|---|
| supply cut: " yes" | 0.83 | 0.44 |
| crude oil will " rise" | 0.65 | 0.47 |
| natural gas will " rise" | 0.44 | 0.61 |

The lift, `p − floor_p`, is what the content itself says. The article moves " yes" to a supply cut from 0.44 to 0.83:
a clear read. For natural gas it moves " rise" *below* its floor, so the article reads as mildly negative for gas
prices, even though 0.44 alone looks like a coin toss. Reading the lift instead of the raw value is the simplest way to
remove question bias, and it is often what makes scores from different questions comparable.

The floor depends only on the question, so it is computed once per call, however many contents share the question.

## From raw probabilities to calibrated scores

A raw `p` depends on the content and on the wording of the question and its answers. To make scores comparable
across questions, sources and years:

1. Score a baseline set of items of the same kind (for example a month of articles from the same sources).
2. Keep, for each question and answer, the distribution of its lift over that baseline.
3. Express each new score as a percentile or z-score of that distribution.

The result is a calibrated probability you can compare and aggregate: per day, per company, per product, per segment.
[Calibration](../calibration.md) covers the details, including `residual`, the probability that went to words other
than your answers, and when to read `p` raw instead of the lift.

## Probabilities you can act on

Once each answer has a probability, the decisions become ordinary code:

- **Thresholds.** Act automatically above one probability, send the band below it to a person. See
  [probabilistic if statements](probabilistic-if.md).
- **Ranking.** Sort a backlog by the probability of "breaks the policy?" and review from the top.
- **Aggregation.** Average probabilities per day or per entity to build indicators, as our own first use did: a daily
  oil-market index read from 15 million news articles.
- **Many questions at once.** Each question costs only its own tokens, so one call can return sentiment, topic,
  urgency and your own checks for the same content.

Accuracy still depends on how the question is written and on a few labelled examples. In our benchmark, adding 8
labelled examples to the prompt took "can a bot resolve this ticket?" from 67% (75% calibrated) to 93% correct on 130
synthetic tickets. See [zero-shot and few-shot classification](zero-shot-few-shot-classification.md) and
[Writing questions](../writing-questions.md).

## Try it

The [quickstart](../quickstart.md) runs the call above against your own server and reads the scores; the
[examples](../examples.md) show several questions about one email with `p` and the floor for each answer. Next:
[probabilistic if statements](probabilistic-if.md) or [Calibration](../calibration.md).
