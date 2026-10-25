# Aspect-based sentiment analysis (ABSA) for reviews and surveys

"Delivery was two days late and the box arrived dented, but support sent a replacement right away and the blender
itself is excellent." One overall label for this review, positive, negative or mixed, loses what matters. Aspect-based
sentiment analysis (ABSA) asks how the customer feels about each aspect separately: negative on delivery and
packaging, positive on support and on the product.

This guide shows how to run aspect sentiment with spinf-serve, how to turn it into numbers you can count, and what it
measured on public and synthetic reviews. The same call runs unchanged against the hosted API at `api.spinf.com` with
a key.

## One template, every aspect

spinf-serve asks questions of the content with templates. A placeholder such as `{aspect}` asks the same question for
every aspect you name, all over one read of the review:

```json
{
  "id": "aspect",
  "template": "\n\nHow does the customer feel about the {aspect} (positive, negative, or not mentioned)?\nAnswer:{?}",
  "options": [" positive", " negative", " not mentioned"],
  "combinations": [
    { "aspect": "delivery" }, { "aspect": "packaging" }, { "aspect": "customer support" },
    { "aspect": "product quality" }, { "aspect": "price" }
  ]
}
```

For each aspect you get the probability of each answer, and they sum to 1. The option "not mentioned" matters: most
reviews talk about two or three aspects, and a forced positive or negative on the others would add noise to every
count. Add an overall question in the same call ("Overall, the review is (positive, mixed, or negative):") and any
other code you need. The review is read once, and each extra question costs only its own tokens.

The aspects are yours: delivery and price for a shop, onboarding and billing for a software product, staff and
cleanliness for a hotel. There is no fixed list and no training run. For reviews and customer feedback there is also a
ready-made starting point, the `spinf/reviews` prompt pack (aspects praised or criticised, implied stars, requests,
risks), which runs in the same call as your own questions ([Prompt packs](../prompt-packs.md)).

## From scores to a feedback dashboard

Probabilities over your own answers add up cleanly:

- **Shares.** The mean of p(negative) for "delivery" over a month of reviews estimates the share of reviews negative
  about delivery, without choosing a cut-off for each review.
- **Segments and trends.** Group by product, region, channel or survey wave, and follow each aspect over time.
- **The reviews behind a number.** Sort by the probability to read the reviews that drive a change.

Each answer also comes with its floor, the same question asked with empty content. With no examples in the prompt,
reading the lift over the floor corrects a question that leans towards one answer. See
[Calibration](../calibration.md).

## Measured on public and synthetic reviews

These figures come from our public benchmark, on the 12B (`spinf/gemma-4-12b-fp8`):

| Task | Test set | Result |
|---|---|---|
| Overall sentiment (3) | 125 synthetic reviews | 96% correct with 8 examples |
| Sentiment per aspect | 625 synthetic aspect labels | 94% correct with 8 examples; 84% with none (calibrated) |
| Polarity | 300 public product reviews | 97% correct with no examples (calibrated) |
| Aspect sentiment | SemEval-2014 restaurants, 1,166 public labels | 89% correct with 4 examples; 83% with none (calibrated) |

On SemEval-2014 the 31B (`spinf/gemma-4-31b-fp8`) reached 90% with 8 examples (one example set). "Examples" are
labelled reviews placed in the prompt, never the reviews being scored; the 12B figures are a mean over three different
example sets. The synthetic reviews were labelled by a language model and checked blind by a second model.

The public datasets are the better guide: synthetic reviews are cleaner than real feedback. Measure on a coded sample
of your own responses before relying on the numbers. The test sets, questions and code are public:
[spinf-benchmarks on GitHub](https://github.com/spinfinc/spinf-benchmarks).

## When to add examples

Overall polarity of a clear review needs no examples. Aspects are harder: the model has to find where the review talks
about each aspect, and decide what counts as, say, "support" in your business. There, a few labelled reviews in the
prompt help most (84% to 94% on the synthetic set). Write them in the same format as the question, cover every answer
at least once, and include reviews that mention an aspect only in passing. See
[Zero-shot and few-shot classification](zero-shot-few-shot-classification.md).

With examples in the prompt, read `p` as it is: the floor is computed without your examples, so it no longer
describes the question the model sees.

## Throughput for every response

The 12B processes about 10,000 tokens per second on one RTX 4090: the content once, then each question's own tokens.
A 120-token answer with the five aspects, an overall question and two more codes is about 350 processed tokens, so a
million survey answers or reviews are 350 million tokens, about 10 hours on one card. That is cheap enough to score
every response instead of a hand-read sample, and to re-score the archive when you add an aspect: the new aspect costs
its own tokens plus one more read of each response.

## Try it

- [Quickstart](../quickstart.md) and [Examples](../examples.md): the aspect grid is the placeholder-and-combinations
  shape of the first call, with "not mentioned" as a third option.
- Related guides: [Coding open-ended survey responses](open-ended-survey-coding.md) and
  [Probabilities from a language model](llm-probabilities.md).
