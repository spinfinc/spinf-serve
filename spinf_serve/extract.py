"""Extraction slots: the model writes the answer, the engine returns the most likely answers with their probabilities.

Design: docs/EXTRACTION.md. A slot is read after `content + template text`; the search
(scorer.Scorer.extract) is a best-first beam over the continuations in which every live beam may also END, with probability
P(beam) x P(end | beam), so "Bud" and "Bud Light" are different answers and P(answer) is the exact probability of that string
followed by an end. Slots:

  {*} text     any tokens but special ones; ends at a token that starts with the template's text after the slot (e.g. a closing
               quote), holds a newline, or is <eos>. The template should open a quote before {*}: '... is "{*}"'.
  {n} number   a number grammar: [space] [- | minus | (] digits [,digits]* [.digits] [ thousand | million | billion | %] [)];
               the answer ends when the grammar is complete and the next token cannot extend it: P(end) = 1 - P(valid extensions).
  {$} currency the number grammar with an optional currency symbol before the digits ($ EUR GBP JPY), after the sign.

A text slot with `generate` (Generate below) is plain text generation instead of the search: greedy (temperature 0) or sampled
(temperature, top_p, top_k, seed), n sequences at once, each ending at the same stops or at max_tokens; the answers come back as
text with a finish reason and no probability (a one-sentence summary has a probability far below MIN_P: the search cannot write it).

Answers are merged over surface forms: text by a normalisation (case, punctuation, English articles, spaces), numbers by value
(and currency). This module holds everything that needs no GPU (grammars, parsing, merging, the response); scorer.py runs the search.
"""

import math
import os
import re
import string
from dataclasses import dataclass, field

SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
CURRENCIES = "$€£¥"  # $ EUR GBP JPY
MINUS = "-−"
NUMBER_ALPHABET = set(" 0123456789.,%()" + MINUS + CURRENCIES)
QUOTES = {'"': '"”', "'": "'’"}  # a closing quote also ends at its typographic twin
BEAM = int(os.environ.get("SPINF_X_BEAM", "6"))  # live beams per step (the env override is for benchmarks)
TOPK = BEAM  # continuations per beam per step
MIN_P = 0.002  # beams (and answers) below this probability are dropped
REL_P = 0.02  # ... and beams below this share of the slot's best answer found so far (they cannot change the answer)
DEFAULT_MAX_TOKENS = {"text": 16, "number": 16, "currency": 16}  # "(1,496.5 million)" is 12 tokens (one per digit)
DEFAULT_TOP_K = 5
GEN_MAX_N = BEAM  # sequences generated at once per slot (they take the slot's beam rows)
GEN_TOP_K_DEFAULT = 40  # sampling: the candidates drawn from when the caller sets none (the end event is always one of them)
GEN_TOP_K_MAX = 100
_PUNCT = set(string.punctuation) | {"’", "‘", "“", "”", "–", "—"}


@dataclass(frozen=True)
class Generate:
    """a text slot generated instead of searched: temperature 0 = greedy (the end event competes with the tokens); above 0 the
    next token is drawn from the top_k candidates (end included) inside the top_p nucleus, with a seeded generator per sequence
    (the same request gives the same text); n sequences at once"""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 0
    n: int = 1


@dataclass
class ExtractPrompt:
    """one extraction read: the tokens after the job's prefix up to the slot (the filled template's text before the slot), the slot
    kind (extract.py) and where a text answer ends; gen: a generated text slot (Generate) instead of the search"""

    key: object
    ids: list[int]
    kind: str  # text | number | currency
    stop: str
    max_tokens: int
    gen: Generate | None = None


@dataclass
class ExtractJob:
    """one input's extraction reads: the content (prefix, read once into a KV cache) and the prompts that decode from it"""

    prefix: list[int]
    media: list  # (offset of the first placeholder, media object) in the prefix
    prompts: list[ExtractPrompt]


# ---------------------------------------------------------------- merging
def norm_text(s: str) -> str:
    """the merge key of a text answer: lower case, punctuation as spaces, no English articles, then no spaces (so "Coca-Cola",
    "Coca Cola", "CocaCola" and "COCA COLA" are one answer)"""
    s = "".join(" " if ch in _PUNCT else ch for ch in s.lower())
    return "".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def parse_value(s: str) -> tuple[float | None, str | None]:
    """a number answer -> (value, currency symbol or None): accounting brackets and minus signs negative, scale words multiply,
    % kept as the number (12.5% -> 12.5)"""
    t = s.strip()
    neg = bool(re.match(r"^[\s$€£¥]*[\-−(]", t))
    cur = next((c for c in CURRENCIES if c in t), None)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None, cur
    v = float(m.group().replace(",", ""))
    rest = t[m.end() :].strip().lower().split()
    if rest and rest[0].strip(").,") in SCALES:
        v *= SCALES[rest[0].strip(").,")]
    return (-v if neg else v), cur


# ---------------------------------------------------------------- the number grammar
class NumberGrammar:
    """a character grammar over token texts; state = (phase, paren, neg, cur, spaced), phases: start pre int comma dot frac post
    closed. "3." is complete (a sentence-final period); "1, 2" ends at 1 (no space after a comma); "(168" is never complete."""

    def __init__(self, currency: bool):
        self.currency = currency
        self.start = ("start", False, False, False, False)

    def _char(self, st, ch):
        ph, paren, neg, cur, spaced = st
        if ch == " ":
            if ph == "start":
                return ("pre", paren, neg, cur, True)
            if ph == "pre" and cur and not spaced:
                return ("pre", paren, neg, cur, True)
            return None
        if ch in MINUS and ph in ("start", "pre") and not neg and not paren:
            return ("pre", paren, True, cur, spaced)
        if ch == "(" and ph in ("start", "pre") and not neg and not paren:
            return ("pre", True, True, cur, spaced)
        if ch in CURRENCIES and self.currency and ph in ("start", "pre") and not cur:
            return ("pre", paren, neg, True, False)
        if "0" <= ch <= "9":
            if ph in ("start", "pre", "int", "comma"):
                return ("int", paren, neg, cur, spaced)
            if ph in ("dot", "frac"):
                return ("frac", paren, neg, cur, spaced)
            return None
        if ch == "," and ph == "int":
            return ("comma", paren, neg, cur, spaced)
        if ch == "." and ph == "int":
            return ("dot", paren, neg, cur, spaced)
        if ch == "%" and ph in ("int", "frac"):
            return ("post", paren, neg, cur, spaced)
        if ch == ")" and paren and ph in ("int", "frac", "post"):
            return ("closed", False, neg, cur, spaced)
        return None

    def step(self, st, text: str):
        """the state after a token's text, or None when the token cannot continue a number here"""
        if st[0] in ("int", "frac") and text.startswith(" ") and text.strip().lower() in SCALES:
            return ("post",) + st[1:]
        for ch in text:
            st = self._char(st, ch)
            if st is None:
                return None
        return st

    @staticmethod
    def complete(st) -> bool:
        return st[0] in ("int", "frac", "post", "closed", "dot") and not st[1]


def number_candidates(token_texts: list[str], special: set[int]) -> list[int]:
    """the token ids that can ever occur in a number answer (a few hundred of the vocabulary)"""
    out = []
    for i, s in enumerate(token_texts):
        if i in special or not s:
            continue
        if set(s) <= NUMBER_ALPHABET or (s.startswith(" ") and s.strip().lower() in SCALES):
            out.append(i)
    return out


def text_stop_ids(token_texts: list[str], stop: str, eos: set[int]) -> list[int]:
    """the tokens that end a text answer: those starting with the template's text after the slot (its first character, and the
    typographic twin of a closing quote), any token holding a newline, and <eos>"""
    firsts = set(QUOTES.get(stop[:1], stop[:1])) if stop else set()
    return sorted({i for i, s in enumerate(token_texts) if (s[:1] in firsts and s) or "\n" in s} | eos)


# ---------------------------------------------------------------- generation
def gen_rngs(gen: Generate):
    """one generator per sequence: the draws of a request are reproducible from its seed (sequence b draws from stream (seed, b))"""
    import numpy as np

    return [np.random.default_rng([gen.seed, b]) for b in range(gen.n)]


def gen_next(gen: Generate, rng, end_lp: float, top_lp, top_ids) -> int | None:
    """the next token of a generated sequence from the step's log-probs: end_lp = log P(the answer ends here: a stop token), top_lp /
    top_ids = the best continuations, best first (at least gen.top_k of them). None = the sequence ends."""
    import numpy as np

    if gen.temperature == 0:
        return None if end_lp >= float(top_lp[0]) else int(top_ids[0])
    k = min(gen.top_k, len(top_ids))
    lp = np.concatenate(([end_lp], np.asarray(top_lp[:k], dtype=np.float64))) / gen.temperature
    order = np.argsort(-lp)  # the end event's position in the ranking depends on its probability like any token's
    p = np.exp(lp[order] - lp[order].max())
    p /= p.sum()
    keep = min(len(p), int(np.searchsorted(np.cumsum(p), gen.top_p, side="left")) + 1)  # the smallest set covering top_p
    p = p[:keep] / p[:keep].sum()
    pick = int(order[int(rng.choice(keep, p=p))])
    return None if pick == 0 else int(top_ids[pick - 1])


# ---------------------------------------------------------------- the response
@dataclass
class Found:
    """the search's completed answers: string -> summed probability (several token paths can spell one string); for a generated
    slot, gen: the sequences (text, finish "stop" | "length") in sequence order"""

    answers: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    generated: int = 0  # tokens run in the decode steps (beams x steps)
    gen: list[tuple[str, str]] = field(default_factory=list)

    def add_gen(self, text: str, finish: str, stop: str) -> None:
        """a generated sequence: stripped, and without the template's stop text when the model wrote it inside its last token
        (e.g. a `."` token ends the sentence after the stop set's `"`-first tokens were masked)"""
        text = text.strip()
        if stop:
            quotes = QUOTES.get(stop[:1], stop[:1])
            while text and text[-1] in quotes:
                text = text[:-1].rstrip()
        self.gen.append((text, finish))

    def add(self, text: str, logp: float) -> None:
        text = text.strip()
        if text:
            self.answers[text] = self.answers.get(text, 0.0) + math.exp(logp)

    def floor(self) -> float:
        """the log-probability a live beam needs to stay: MIN_P, or REL_P of the best answer found so far"""
        best = max(self.answers.values(), default=0.0)
        return math.log(max(MIN_P, REL_P * best))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z > -700 else 0.0


def result(kind: str, found: Found, top_k: int, calibration=None, gen: Generate | None = None) -> dict:
    """merged answers, best first: text by norm_text, numbers by (value, currency); each with its summed probability. `mass` = the
    probability on all answers found (the rest went to answers below MIN_P or to forms that do not end). For numbers,
    `p_negative` = the share of the top value's magnitude that is on its negative (the sign is often ambiguous in accounting
    tables). `p_calibrated` = sigmoid(a logit(p) + b) of the top answer when the caller sends a calibration. A generated slot:
    its n sequences as {text, finish}, no probability."""
    if gen is not None:
        return {"answers": [{"text": t, "finish": f} for t, f in found.gen]}
    groups: dict[object, dict] = {}
    for text, p in sorted(found.answers.items(), key=lambda x: -x[1]):
        if kind == "text":
            key = norm_text(text) or text
            g = groups.setdefault(key, {"text": text, "p": 0.0})
        else:
            v, cur = parse_value(text)
            if v is None:
                continue
            key = (round(v, 9), cur if kind == "currency" else None)
            g = groups.setdefault(key, {"text": text, "p": 0.0, "value": v} | ({"currency": cur} if kind == "currency" else {}))
        g["p"] += p
    answers = sorted(groups.values(), key=lambda g: -g["p"])
    out: dict = {"answers": [{**g, "p": round(g["p"], 6)} for g in answers[:top_k]], "mass": round(sum(found.answers.values()), 6)}
    if kind != "text" and answers:
        mag = abs(answers[0]["value"])
        same = [g for g in answers if abs(abs(g["value"]) - mag) <= 1e-9 * max(1.0, mag)]
        tot = sum(g["p"] for g in same)
        out["p_negative"] = round(sum(g["p"] for g in same if g["value"] < 0) / tot, 6) if tot > 0 else None
    if calibration is not None and answers:
        p = min(max(answers[0]["p"], 1e-9), 1 - 1e-9)
        out["p_calibrated"] = round(_sigmoid(calibration.a * math.log(p / (1 - p)) + calibration.b), 6)
    return out
