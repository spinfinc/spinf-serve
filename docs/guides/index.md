# Guides

spinf-serve returns the probability of each answer you allow, for every question you ask about your content, from one
read of that content on your own GPU. These guides explain the ideas behind that (probabilities instead of generated
text, the empty-content floor, thresholds, few-shot examples) and how to apply them to common jobs, with the measured
figures behind each claim.

## Concepts

- [Decision models](decision-models.md): an open-weights model as a calibrated decision model, at scale: your options, a
  probability for each, no generated text.
- [Probabilities from a language model](llm-probabilities.md): confidence scores over the answers you allow, the
  empty-content floor, and calibrated scores you can compare.
- [Probabilistic if statements](probabilistic-if.md): decisions in code on confidence thresholds, with a band for
  human review, chosen on your own history.
- [Zero-shot and few-shot classification](zero-shot-few-shot-classification.md): classification with no
  fine-tuning, and what 4 or 8 labelled examples in the prompt change, measured.
- [Scoring without output tokens](no-output-tokens.md): why scoring costs less than generating, and what a million
  documents take when the content is read once for every question.

## Jobs

- [Entity-level sentiment for news](entity-sentiment-analysis.md): sentiment toward each company an article names,
  aggregated into daily indicators.
- [Aspect-based sentiment analysis](aspect-based-sentiment-analysis.md): sentiment per aspect for reviews and
  survey answers, measured on SemEval-2014.
- [Coding open-ended survey responses](open-ended-survey-coding.md): your codebook as questions, every verbatim
  coded.
- [Custom content moderation policies](custom-content-moderation.md): your rules as questions, a few examples,
  thresholds per category, next to the official moderation pack.
- [Intent classification and routing](intent-classification.md): intents as answer options, thresholds for
  automation, and many intents in two steps.

## Reference

The [documentation](../README.md) covers the engine and its API: the [quickstart](../quickstart.md), the
[scoring API](../scoring-api.md), [writing questions](../writing-questions.md) and [calibration](../calibration.md).
The benchmark behind the figures quoted in these guides is public:
[spinf-benchmarks](https://github.com/spinfinc/spinf-benchmarks).
