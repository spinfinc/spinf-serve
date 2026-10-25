# Entity-level sentiment analysis for financial news

A news article rarely says one thing about one company. A supplier's shortage is bad news for the carmaker and may be
good news for its competitor, and one article can name five companies. Entity-level sentiment analysis, also called
targeted sentiment, asks about each company separately: is this article good or bad news for *this* company?

This guide shows how to run it with spinf-serve, how to turn the scores into a daily sentiment index, and what it
measured on our benchmark and in production. The same call runs unchanged against the hosted API at `api.spinf.com`
with a key.

## One read, a question per company

spinf-serve asks questions of the content with templates. A placeholder such as `{company}` asks the same question for
every name you list, all over one read of the article ([Scoring API](../scoring-api.md), `combinations`):

```json
{
  "id": "sentiment",
  "template": "\n\nIs this news positive, neutral or negative for {company}?\nAnswer:{?}",
  "options": [" positive", " neutral", " negative"],
  "combinations": [ { "company": "Northwind Motors" }, { "company": "Voltra" } ]
}
```

For each company you get the probability of each answer, and they sum to 1. A template takes up to 1,000 values, so
one call can follow a whole watchlist. The same article can carry other questions too:

- **Events:** "Does it report a supply problem?", "A guidance change?", "A lawsuit?", as yes/no questions.
- **Impact:** "How much could this news move the share price of {company} (none, small, large)?"
- **Topic:** earnings, supply chain, regulation, deals, management.
- **Open fields:** the names the article mentions, when you do not have a list, with an extraction slot such as
  `The company this news is about is "{*}"` ([extraction slots](../scoring-api.md#extraction-and-generation-slots)).

The article is read once per call, and each extra question costs only its own tokens. That is what makes hundreds of
questions per article practical.

## From scores to a sentiment index

A single article score is noisy. The value for trading research and alternative data comes from aggregating many of
them:

1. **Remove the question's bias.** Each answer comes with its floor, the same question asked with empty content.
   The lift over the floor is what the article itself says.
2. **Place each score on its own history.** Rank or standardise the lift against the same question over a baseline of
   articles from the same sources, so different questions and entities are comparable.
3. **Aggregate** per day, per entity and per source into indicators, then compare them with prices, flows or outcomes.

[Calibration](../calibration.md) covers these steps. The result is a measured read of what the news says. It is not
a forecast.

## In production: a daily oil-market index

Our own first use, a daily oil-market index read from 15 million news articles, is built on these reads. It asks
about five hundred fixed questions of every oil article and aggregates the answers into a daily read of the press:

- **15M** news articles over five years, 2022–2026
- **~1T** tokens processed across research iterations
- **0.85** correlation with crude in the best year, **0.80** pooled on WTI
- **92%** of big 20-day moves read in the right direction

## Measured on the benchmark

On 108 synthetic news articles about fictional companies (255 company mentions), labelled by a language model and
checked blind by a second model, the 12B (`spinf/gemma-4-12b-fp8`) scored:

- **95% correct sentiment per company** with 8 labelled example articles in the prompt, and 76% with no examples.
- **98% correct** on "does it report a supply problem?" with 8 examples (94% with none).
- **99% correct** on "does it report a guidance change?" with 8 examples (98% with none).
- **94% correct main topic** out of 5 with 8 examples (88% with none, calibrated). The 31B (`spinf/gemma-4-31b-fp8`)
  reached 95% with no examples.

Synthetic articles are cleaner than real news. On 300 public financial-news posts, which are short and informal,
sentiment accuracy was 69% with the 12B and 82% with the 31B, with the same 8 examples in the prompt. Test on a sample
of your own sources before relying on the numbers. The test sets, questions and code are public:
[spinf-benchmarks on GitHub](https://github.com/spinfinc/spinf-benchmarks).

## Throughput at archive scale

The 12B processes about 10,000 tokens per second on one RTX 4090: the content once, then each question's own tokens.
An 800-token article with 20 questions of about 25 tokens each is about 1,300 processed tokens, so a million articles
are 1.3 billion tokens, about 36 hours on one card, and a question added to the call costs only its own tokens. Send
the archive in large calls: many articles per call, so the engine packs them into full passes.

## Practical notes

- **Use the names as they appear.** Put the company name in the placeholder the way the articles write it, and keep
  the list per article to the names it mentions or your watchlist.
- **Prefer simple questions for events.** Yes/no questions about what an article reports give the clearest reads.
- **Keep the questions fixed.** Scores compare over time only under the same template, answers and model build; the
  response's `system_fingerprint` tells you when the build changed.
- **Send batches.** Many articles per call fill the engine's passes and share the floors of the shared questions.

## Try it

- [Quickstart](../quickstart.md): the first call, with a placeholder over two commodities. [Examples](../examples.md) has
  the many-contents-per-call shape.
- Related guides: [Probabilities from a language model](llm-probabilities.md) and
  [Zero-shot and few-shot classification](zero-shot-few-shot-classification.md).
