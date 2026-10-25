# Scoring without output tokens: the cost of classifying millions of documents

When a language model classifies by generating an answer, you pay for the text it writes, wait while it writes it,
and then parse it. For a label, a yes/no or a score, the generated text is overhead. spinf-serve classifies by scoring
instead: it returns the probability of each answer you list and generates nothing for a scored question. There are no
output tokens; the work is the input, read once. (Extraction and generation slots exist for open fields, and they do
add decode steps: [Scoring API](../scoring-api.md#extraction-and-generation-slots).) The same request runs unchanged
against the hosted API at api.spinf.com with a key; the figures below are for the engine on your own card.

## What generation costs in a classification job

A generated answer adds three costs that grow with every document:

- **Output tokens.** Even a short reply ("Category: billing. Confidence: high.") is several tokens, and explanations
  run to hundreds. Each one is a decode step, produced one at a time.
- **Parsing.** Free text has to be mapped back to your labels, with rules for replies that do not match any of them.
- **Waiting.** Text is produced one token at a time after the input has been read.

A scored answer has none of these. The response lists your answers with their probabilities; the `choices` of a chat
completion are empty because nothing is written.

## What the engine processes

The work of a call is its processed tokens, which the response reports in `usage`:

- **the content, once per call**, however many questions you ask about it;
- **each question's own tokens**: the filled template and its answers;
- **one empty floor per question per call**: the same question scored with no content, which gives the question's
  bias ([Calibration](../calibration.md)). You can skip it with `"empty_floor": false` once you have it.

The 12B (`spinf/gemma-4-12b-fp8`) processes about 10,000 of these tokens per second on one RTX 4090, with inputs
packed together into 16k-token passes. Batch work sends many documents per call with `inputs`: the floors are then
shared and the packing is dense. The server's limits are in [Configuration](../configuration.md).

## The time for a million documents

Take documents of about 300 tokens, each asked 10 questions of about 20 tokens:

| Step | Tokens |
|---|---|
| content | 300 tokens |
| + 10 questions × 20 tokens | 200 tokens |
| = tokens per document | about 500 |
| × 1,000,000 documents | 500M tokens |
| ÷ 10,000 tokens per second | **about 14 hours on one RTX 4090** |

With many documents per call, the floors add almost nothing per document. The same arithmetic gives about 40 hours for
a million 800-token news articles with 20 questions each (1,200 tokens a document), and about 8 hours for a million
120-token survey answers with 8 questions (280 tokens a document). A second card halves each figure; the 31B on a
48 GB card runs at roughly a third of the 12B's rate.

## Many questions cost little

Because the content is read once per call, the cost per question is only its own tokens. Asking 10 questions of
a document costs far less than 10 separate classifications of it: the 300 tokens of content are paid once, not ten
times. This changes how jobs are designed:

- ask the questions you would otherwise skip, such as a policy check next to the topic and sentiment;
- ask the same question for every company, product or aspect with a placeholder, up to 1,000 values per question;
- re-score an archive when the codebook or the questions change, instead of living with the first version.

## Labelled examples are input too

Few-shot examples placed in the prompt raise accuracy the most ([zero-shot and few-shot
classification](zero-shot-few-shot-classification.md)), and they are part of the content, so they are read with every
document. In our ticket benchmark, 8 short labelled examples added about 1,100 tokens per ticket: the per-document
work goes from about 500 tokens to about 1,600, so a million tickets take about 44 hours on one card instead of 14.
Keep the examples short, and add them only to the questions that need them.

## Batch processing at scale

The engine is built for throughput: backfills, archives and periodic batches, with many inputs per call and one
process per GPU. Our own first use, a daily oil-market index, was built this way on 15 million news articles and about
a trillion tokens across research iterations. A live flow runs the same questions one item at a time against the same
server ([configuration guide](../configuration.md#throughput)).

## Try it

The [examples](../examples.md) send many contents in one call and show the `usage` block that counts the work; the
[quickstart](../quickstart.md) is the first call. Related: [Configuration](../configuration.md) and
[probabilities from a language model](llm-probabilities.md).
