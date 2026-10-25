# Coding open-ended survey responses with a language model

Open-ended questions are where respondents say what the closed questions missed: why they gave a low NPS score, what
went wrong with a delivery. They are also the most expensive part of a survey to analyse: coding the verbatims by hand
means reading every answer against a codebook, so most teams code a sample.

spinf-serve runs an open-weights language model as a scoring engine for exactly this kind of job: the same questions
asked of every response, with a probability for each answer you allow. This guide shows how to turn a codebook into
questions, check the result against your own coders, and re-code an archive when the coding frame changes. The same
call runs unchanged against the hosted API at `api.spinf.com` with a key.

## From codebook to questions

A codebook (or coding frame) is a list of codes with a definition for each. With spinf-serve, each code becomes a
closed question, and the answers are the words you want back:

- **One yes/no question per code** when a response can carry several codes: "Does the customer mention a late or
  missed delivery?", "Does the customer mention the price?", "Does the answer suggest a new feature?".
- **One closed question for a single-choice code**, such as the main theme or the overall tone: "Overall, the answer
  is (positive, mixed, or negative):".
- **A template with a placeholder** for codes that repeat across topics. One template, "How does the customer feel
  about the {aspect} (positive, negative, or not mentioned)?", asks the same question for delivery, packaging, support,
  quality and price in one call. See the [aspect-based sentiment guide](aspect-based-sentiment-analysis.md).

Put the definition of the code in the question, the way you would write it in the codebook. Start each response with a
short label, such as `Survey answer:` or `NPS comment:`, so the model knows what it is reading.

```json
{
  "id": "delivery_issue",
  "template": "\n\nDoes the customer mention a late, missed or damaged delivery?\nAnswer:{?}",
  "options": [" yes", " no"]
}
```

For each response and each code, spinf-serve returns the probability of each answer you listed; nothing is generated.
When a code needs an open answer (the product named, an amount), an
[extraction slot](../scoring-api.md#extraction-and-generation-slots) returns the most likely answers with their
probabilities, in the same call.

## Teach the subtle codes with examples

Some codes are obvious ("mentions the price"); others depend on how your team draws the line ("complaint vs
suggestion"). For those, put a few coded responses in the content before the one to score: 4 to 8 examples, covering
each answer and the cases your coders disagree on. No training run is needed.

In our public benchmark, examples made the largest difference on per-aspect sentiment: 94% correct with 8 examples,
against 84% with no examples (calibrated), on 625 labels from 125 synthetic reviews. Overall sentiment (positive,
mixed or negative) was 96% correct with 8 examples on the same reviews. The test sets and the code are on
[GitHub](https://github.com/spinfinc/spinf-benchmarks). Synthetic answers are cleaner than real ones, so measure on your
own data, as described below.

## Check agreement with your coders

Before coding a whole survey, measure spinf-serve the way you would measure a new human coder:

1. Take a sample of responses that your team has already coded: 100 to 300 is usually enough.
2. Score them with your questions, with and without examples, and with two or three wordings of the subtle codes.
3. For each code, compare with your coders: agreement, and the codes where the two disagree. Read the disagreements;
   they often show a code whose definition was ambiguous for people too.
4. Keep the version that agrees best, and choose a threshold per code (for example, count a code when its probability
   is above 0.5, and send answers between 0.4 and 0.6 to a person).

A round on 300 responses is about 100,000 processed tokens, a few seconds on one card, so trying several wordings
costs nothing but the writing. Once you are happy, keep the questions, answers and examples fixed, so that results
compare from one wave to the next.

## Count, segment and track

Because every code is a probability over your own answers, the results add up cleanly: the share of responses that
mention delivery, per segment, per wave, per product. You can count with a threshold, or sum the probabilities to get
an expected count that does not depend on where you put the cut. Each answer also comes with its floor, the same
question asked with no response, so different codes can be compared; see [Calibration](../calibration.md).

## Re-code the archive when the codebook changes

Codebooks change: a new theme appears, two codes merge, a definition is tightened. With manual coding, the old waves
keep the old frame. With spinf-serve, re-scoring the archive is a batch job: send past responses with the new
questions, many answers per call, and the whole history is coded the same way.

The job stays small because each extra question costs only its own tokens. The 12B processes about 10,000 tokens per
second on one RTX 4090; with 120-token answers and 8 questions, a response is about 350 processed tokens in batched
calls, so 1 million responses are about 350 million tokens, about 10 hours on one card.

Any text works, from survey verbatims to NPS comments and app store reviews, and with your own server the responses
never leave your machine.

## Try it

[Quickstart](../quickstart.md) for the first call, [Examples](../examples.md) for many responses in one call, and
[Writing questions](../writing-questions.md) for wording and examples. Related:
[aspect-based sentiment analysis](aspect-based-sentiment-analysis.md) and
[zero-shot and few-shot classification](zero-shot-few-shot-classification.md).
