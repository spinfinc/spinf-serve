# Writing questions

As with any language model, how you write the questions changes the results, often by more than the choice of model.
The good news: scoring is cheap enough to try several wordings, and a few labelled examples, on a few hundred items of
your own data in minutes, and to keep the version that works best. This page collects what made the biggest difference
in our benchmarks (Gemma 4 12B, synthetic and public datasets). The test sets, the exact questions and example sets,
and the code to re-run them are public: [spinf-benchmarks on GitHub](https://github.com/spinfinc/spinf-benchmarks).

## 1. Say what the content is

Start the content with a short label that says what follows: `Customer support message:`, `User post:`,
`Customer review:`, `News article:`. It tells the model how to read the text. In our moderation benchmark, dropping the
label `User post:` took the zero-shot policy category from 79% to 52% correct.

```json
{ "role": "user", "content": "Customer support message:\nHi, since this morning's SSO change nobody on our team can log in…" }
```

## 2. Name the answers in the question

A closed question works best when the question lists its answers, and the options are those words:

```json
{
  "id": "topic",
  "template": "\n\nTopic (billing, technical, account, shipping, or cancellation):{?}",
  "options": [" billing", " technical", " account", " shipping", " cancellation"]
}
```

- One or two words per answer, clearly different from each other, each with a leading space.
- For yes/no questions, put the definition in the question: `"Is this urgent (business blocked, security, legal
  threat, data loss, or a deadline within 24 hours)?\nAnswer:{?}"`.
- Keep the list manageable. A few to a few dozen answers work well; for a long list (e.g. 77 banking intents), ask
  for the broad category first, then the answer within it.
- Try two or three wordings on items you know the answer to, and keep the one that separates them best.

## 3. Add a few labelled examples

Labelled examples placed in the content, before the item to score, are the biggest lever: they show the model your
labels, your boundaries and your edge cases. With 8 examples in the prompt, our benchmarks went from:

| Question | No examples | 8 examples |
|---|---|---|
| Ticket: can a bot resolve it? | 67% | 93% |
| Moderation: policy category (7) | 79% | 98% |
| Review: sentiment per aspect | 43% | 94% |
| Public prompt-injection set | 78% | 91% |

Write each example in the same format as the question, with its answer, and separate them from the item:

```text
Customer support message:
I was charged twice for September, please refund one of the charges.

Topic (billing, technical, account, shipping, or cancellation): billing

---

Customer support message:
The export button does nothing since the last update.

Topic (billing, technical, account, shipping, or cancellation): technical

---

Customer support message:
<the ticket to score>
```

- 4 to 8 examples are usually enough; cover every answer at least once, and include the cases people get wrong.
- Never score the items you used as examples, when you measure accuracy.
- Examples are part of the content, so they are read with every input: 8 short examples add about 1,100 tokens per
  item. Send many inputs per call with `inputs`, so the questions' floors are computed once.

## 4. Raw or calibrated?

The floor (`floor_p`, the same question with empty content) removes the question's own bias; see
[Calibration](calibration.md).

- **Without examples**, read the lift over the floor for multi-choice questions: it often fixes a question that leans
  towards one answer (ticket topic: 75% raw, 87% calibrated; aspects: 43% raw, 84% calibrated).
- **With examples**, read `p` as it is: the floor is computed without your examples, so it no longer describes the
  question the model sees (aspects with 8 examples: 94% raw, 80% calibrated).

## 5. Iterate on your own data

1. Label 100 to 300 items of your own data (fewer if they are easy to tell apart).
2. Score them zero-shot, then with 4 and 8 examples, with two or three wordings of each question.
3. Keep the version that separates your labels best, pick your thresholds, and keep the template, the answers and the
   examples fixed from then on: scores compare over time only under the same question and `system_fingerprint`.

A round on 300 items with 8 examples is about 400,000 tokens: under a minute on one 4090.

## 6. Ask many questions at once

Each question costs only its own tokens: the content is read once per call. So add the checks you would otherwise skip
to the same call: route a message by intent, check it for prompt injection and against your content policy, all in
one read. Use placeholders (`{aspect}`, `{company}`) with `combinations` to ask the same question for many values. When
the answer is open (a name, an amount, a date), add an extraction slot ([Scoring API](scoring-api.md#extraction-and-generation-slots))
to the same list.

## 7. Writing extraction templates

An extraction slot (`{*}` text, `{n}` number, `{$}` amount: [Scoring API](scoring-api.md#extraction-and-generation-slots))
is written like a question, and the wording decides how much of the model's probability lands on complete, parseable
answers. What the maintainers measured (on a set of TV commercials, Imagewoof dog photos, SQuAD, Natural Questions and
TAT-QA financial reports, Gemma 4 12B and 31B):

- **Text `{*}`: put the slot in quotes.** `The brand featured in this commercial is "{*}"`. The opening quote tells the
  model a name or a short span comes next, and the closing quote is where the answer ends. Against a colon ending
  (`Brand: {*}`) it doubled the probability on complete answers (the brand from 32 frames: `mass` 0.39 to 0.71) and
  calibrated better. Without quotes the model also writes formatting: `Company: {*}` answered `<strong>Apple</strong>`
  at `p` 0.01.
- **Make the slot unambiguous.** `The company is "{*}"` reads as "the company is [an adjective]": the top answers were
  "on track" and "thrilled". `The company in this news is "{*}"` gives "Apple" (0.20) and "Apple Inc" (0.08). Name the
  thing and where it is ("featured in this commercial", "in this news").
- **Numbers `{n}` / `{$}`: no quotes, no filler, end right at the slot.** `\nQ: How many units were sold?\nA:{n}` or
  `\nUnits sold: {n}`. Inside quotes the model wants words and units, and forcing digits throws the probability away. A
  filler before the slot ("The answer is {n}") cost 1 to 5 points of accuracy and a third of the mass.
- **One number slot, not integer or decimal variants.** An integer-only grammar lost even on integer answers (SQuAD
  writes "3.5 million": 0.87 against 0.95 right with the full grammar); a decimal grammar without scale words and `%`
  lost 8 points. Check whether a value is a whole number on `value`.
- **Use `{$}` on financial content.** Allowing a currency symbol kept more probability on the answer in financial
  reports (`mass` 0.52 against 0.32, and a better ranking) and was neutral elsewhere. The symbol comes back as
  `currency`, or `null` when the model wrote none (when "$" is already in the content, say).
- **Signs are uncertain in accounting tables.** On TAT-QA negatives the magnitude was right 92 to 93 % of the time but
  the sign only 70 %: brackets are often presentation ("(168)" for an allowance). Report the sign from `p_negative`, not
  from the top answer.
- **Introduce media with a line of text.** Before a video: "A 31-second TV commercial:"; the question then refers to
  "this commercial". The same for an image or an audio clip.
- **Scored or extracted?** When the answers are a known short list, a scored question (`{?}` with options) is cheaper,
  has an empty floor and gives a probability over exactly that list. Extract when the set is open (brands, names,
  amounts). Both run in one call, on one read of the content.
- **Extraction does not disturb the scored questions.** Adding slots to a call moves the scored reads only by the
  packing noise of the scheme, the same as adding more scored questions; on the batch-invariant `fp8s` scheme it does
  not move them at all.

Reading an extraction's `p`, and calibrating it with a few labels: [Calibration](calibration.md#extraction-probabilities).

## 8. Generated text

The search behind a slot finds short answers: a sentence of twenty words has a probability far below the search's floor
(0.2 %) and is never found. For a sentence or a caption, set `"generate": true` on a `{*}` slot and the model writes the
text instead, in the same call and the same read of the content; the answer comes back as text with a finish reason and
no probability. Greedy decoding is the default and the right choice for a metadata field; sampling at the model's own
defaults cost 4 F1 points on a one-sentence description, at temperature 0.7 one point.

Write the lead-in for the model you serve. On a base model (the Gemma checkpoints) open a quote, zero-shot:
`Summary of the commercial in one sentence, starting with the advertiser's name: "{*}"` (few-shot examples get
copied). On a chat model (the Qwen checkpoint) put the instruction in the user turn and open the slot in the assistant
turn. Measured by the maintainers on a set of TV commercials with a one-sentence description as gold (16 frames at
`low` plus the transcript, greedy): the Qwen model named the brand in 96 % of the sentences and the product in 90.5 %
(token F1 0.47 against the gold sentence), the 12B 92.5 % and 80.5 % (F1 0.46); a sentence is 27 to 35 generated tokens,
and none was cut at the 64-token cap.
