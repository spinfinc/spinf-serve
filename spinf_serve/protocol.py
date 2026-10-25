"""The scoring protocol as the engine receives it (the public request shape: docs/ARCHITECTURE.md).

A front end (the hosted API, a gateway) may forward a request with `messages` normalised to `inputs` and its own limits. This
module parses it and runs the checks that need no tokenizer (a front end may run the same ones first; the engine repeats them so
it never depends on the caller). Token checks and template expansion are in compiler.py.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from spinf_serve import extract as ex

MAX_PLACEHOLDERS = 5
MAX_OPTIONS = 1000
MAX_OPTION_TOKENS = 5
MAX_ALL_COMBINATIONS = 1000
MAX_VALUE_CHARS = 256
SLOT = "{?}"
# extraction slots (the model writes the answer): {*} text, ended by the template's text after the slot (e.g. a closing quote) or a
# newline; {n} a number and {$} an amount, ended by the number grammar (spinf_serve.extract). Only {*} may have text after it.
EXTRACT_SLOTS = {"{*}": "text", "{n}": "number", "{$}": "currency"}
MAX_EXTRACT_TOKENS = 64  # a searched slot's max_tokens (6 beams)
MAX_GENERATE_TOKENS = 100  # a generated slot's: up to 100 decode steps
MAX_EXTRACT_ANSWERS = 20
MAX_FRAMES = 1024
_NAME = re.compile(r"[A-Za-z0-9_]+")


class Rejected(Exception):
    """a request the engine refuses before any GPU work: (code, message, param) as in the error envelope"""

    def __init__(self, code: str, message: str, param: str | None = None):
        super().__init__(message)
        self.code, self.message, self.param = code, message, param

    def body(self) -> dict:
        return {"code": self.code, "message": self.message, "param": self.param}


class ImageURL(BaseModel):
    url: str
    detail: str = "auto"


class InputAudio(BaseModel):
    data: str
    format: str


class VideoFrame(BaseModel):
    t: float = Field(ge=0, description="seconds from the start of the video")
    image_url: ImageURL


class Video(BaseModel):
    """frames sampled by the caller (the engine never decodes video): each is read as an image at `detail` (default low = 70
    tokens), preceded by its timestamp as text, as Gemma's own video prompt ("mm:ss <image> mm:ss <image> ...")"""

    frames: list[VideoFrame]
    detail: str = "low"


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None
    input_audio: InputAudio | None = None
    video: Video | None = None


class Message(BaseModel):
    role: str = "user"
    content: str | list[ContentPart]


class Calibration(BaseModel):
    """a caller's correction of an extraction's raw probability: p_calibrated = sigmoid(a * logit(p) + b)"""

    a: float = 1.0
    b: float = 0.0


class Generate(BaseModel):
    """a text slot ({*}) generated instead of searched (extract.Generate): `generate: true` = greedy; temperature above 0 samples
    the next token from the top_k candidates (default 40; the end of the answer is always one of them) inside the top_p nucleus,
    reproducibly from `seed`; `n` sequences at once (at most the slot's beam rows). Answers come back as text with a finish
    reason ("stop" at the template's stop text, a newline or end of text; "length" at max_tokens), without a probability."""

    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int | None = Field(None, ge=1, le=ex.GEN_TOP_K_MAX)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    n: int = Field(1, ge=1, le=ex.GEN_MAX_N)

    def spec(self) -> ex.Generate:
        top_k = self.top_k or (ex.GEN_TOP_K_DEFAULT if self.temperature > 0 else 1)
        return ex.Generate(self.temperature, self.top_p, top_k, self.seed, self.n)


class Query(BaseModel):
    id: str = Field(max_length=128)
    template: str
    options: list[str] | None = None  # scored queries ({?}); extraction queries ({*} {n} {$}) have none
    combinations: list[dict[str, str]] | Literal["all"] | None = None
    values: dict[str, list[str]] | None = None
    empty_floor: bool | None = None
    # a named query group (e.g. a prompt pack: "spinf/moderation@0.1.3"): per input, the queries without a group come first, as
    # always; each named group follows in its own token trie, starting on a 128-token block, in name order. A group's scores then
    # do not depend on the other queries of the call (a pack's scores are the same with or without the caller's own queries), and
    # the ungrouped queries score as they would without the groups
    group: str | None = Field(None, max_length=128)
    # extraction only: max_tokens up to MAX_EXTRACT_TOKENS searched, MAX_GENERATE_TOKENS generated (checked below)
    max_tokens: int | None = Field(None, gt=0, le=MAX_GENERATE_TOKENS)
    top_k: int | None = Field(None, gt=0, le=MAX_EXTRACT_ANSWERS)
    calibration: Calibration | None = None
    # a text slot generated instead of searched: true (greedy) or the sampling settings (Generate)
    generate: bool | Generate | None = None

    @model_validator(mode="after")
    def _max_tokens_cap(self):
        if self.max_tokens is not None and self.max_tokens > MAX_EXTRACT_TOKENS and not self.generate:
            raise ValueError(f"max_tokens: a searched slot is at most {MAX_EXTRACT_TOKENS} tokens ({MAX_GENERATE_TOKENS} generated).")
        return self

    def gen_spec(self) -> ex.Generate | None:
        if not self.generate:
            return None
        return ex.Generate() if self.generate is True else self.generate.spec()


class Input(BaseModel):
    id: str = Field(max_length=128)
    messages: list[Message]
    query_ids: list[str] | None = None
    queries: list[Query] | None = None
    # the values of the packs' parameters for this input (a pack with `parameters`, e.g. the brand of interest): substituted into
    # the pack's context only, never into the shared queries; the server checks them against the packs of the call
    parameters: dict[str, str | int | float | bool] | None = None


class Scoring(BaseModel):
    queries: list[Query] = Field(default_factory=list)
    empty_floor: bool = True
    case_insensitive: bool = True


class Limits(BaseModel):
    """the request's limits: a server's settings, or what a trusted front end sends per call (capped by the server's); every check
    before GPU work reads them here"""

    max_content_tokens: int = Field(4096, gt=0)
    max_query_tokens: int = Field(256, gt=0)
    max_prompts_per_input: int = Field(10_000, gt=0)
    max_processed_tokens: int = Field(2_000_000, gt=0)
    # extraction slots ({*} {n} {$}: queries x combinations) per input; the API bills them per slot
    max_extractions_per_input: int = Field(10, ge=0)
    # the constants of the hosted service, settable on a server of one's own
    max_options: int = Field(MAX_OPTIONS, gt=0)
    # an option's tokens: a cost guard, not a model limit (one probability per long string tells little; extraction is the tool)
    max_option_tokens: int = Field(MAX_OPTION_TOKENS, gt=0)
    max_frames: int = Field(MAX_FRAMES, gt=0)  # per video
    max_audio_seconds: int = Field(600, gt=0)
    max_image_pixels: int = Field(50_000_000, gt=0)

    def capped(self, by: "Limits") -> "Limits":
        """these limits, none above `by` (a trusted caller's request against the server's settings)"""
        return Limits(**{k: min(getattr(self, k), getattr(by, k)) for k in Limits.model_fields})


class ScoreRequest(BaseModel):
    model: str
    inputs: list[Input] = Field(min_length=1)
    scoring: Scoring = Field(default_factory=Scoring)
    limits: Limits = Field(default_factory=Limits)


# ---------------------------------------------------------------- templates
class Template:
    """a parsed template: literal text and placeholder names up to the slot. The scored slot {?} ends the template; an extraction
    slot ({*} {n} {$}) too, except {*}, whose trailing text (e.g. the closing quote) is where the answer ends (self.stop)."""

    def __init__(self, text: str, param: str):
        parts: list[tuple[str, str]] = []  # ("text", s) | ("name", n)
        i, buf, slot = 0, [], False
        self.kind, self.stop = "score", ""
        for mark, kind in EXTRACT_SLOTS.items():
            if mark in text.replace("{{", "").replace("}}", ""):
                at = _find_slot(text, mark)
                if at is None:
                    continue
                after = text[at + len(mark) :]
                if kind != "text" and after:
                    raise Rejected("invalid_template", f"the slot {mark} must end the template.", param)
                if "{" in after.replace("{{", "").replace("}}", "") or "\n" in after:
                    msg = f"the text after {mark} is where the answer ends: no placeholders or newlines."
                    raise Rejected("invalid_template", msg, param)
                if SLOT in text:
                    raise Rejected("invalid_template", "a template has one slot: {?} or an extraction slot.", param)
                self.kind, self.stop = kind, after.replace("{{", "{").replace("}}", "}")
                text = text[:at] + SLOT
                break
        while i < len(text):
            if text.startswith("{{", i):
                buf.append("{")
                i += 2
            elif text.startswith("}}", i):
                buf.append("}")
                i += 2
            elif text.startswith(SLOT, i):
                if i + len(SLOT) != len(text):
                    raise Rejected("invalid_template", "the scored slot {?} must end the template.", param)
                slot = True
                i += len(SLOT)
            elif text[i] == "{":
                j = text.find("}", i)
                name = text[i + 1 : j] if j > 0 else ""
                if j < 0 or not _NAME.fullmatch(name):
                    raise Rejected("invalid_template", f"bad placeholder at character {i}: use {{name}}, {{?}} or {{{{ }}}}.", param)
                if buf:
                    parts.append(("text", "".join(buf)))
                    buf = []
                parts.append(("name", name))
                i = j + 1
            elif text[i] == "}":
                raise Rejected("invalid_template", f"unmatched }} at character {i} (a literal brace is }}}}).", param)
            else:
                buf.append(text[i])
                i += 1
        if buf:
            parts.append(("text", "".join(buf)))
        if not slot:
            raise Rejected("invalid_template", "the template needs the scored slot {?} at its end.", param)
        names = list(dict.fromkeys(n for kind, n in parts if kind == "name"))
        if len(names) > MAX_PLACEHOLDERS:
            raise Rejected("invalid_template", f"{len(names)} placeholders; at most {MAX_PLACEHOLDERS} besides {{?}}.", param)
        self.parts = parts
        self.names = names

    def fill(self, values: dict[str, str]) -> str:
        return "".join(v if kind == "text" else values[v] for kind, v in self.parts)


def _find_slot(text: str, mark: str) -> int | None:
    """the position of an extraction slot mark outside {{ }} escapes (None when only escaped)"""
    i = 0
    while i < len(text):
        if text.startswith("{{", i) or text.startswith("}}", i):
            i += 2
        elif text.startswith(mark, i):
            return i
        else:
            i += 1
    return None


def combinations(q: Query, tpl: Template, param: str) -> list[dict[str, str]]:
    """the placeholder values to score, checked against the template"""
    names = set(tpl.names)
    if q.combinations == "all":
        if not q.values:
            raise Rejected("missing_value", '"all" needs `values`: {name: [value, ...]}.', f"{param}.values")
        if set(q.values) != names:
            raise _mismatch(set(q.values), names, f"{param}.values")
        n = 1
        for name in tpl.names:
            if not q.values[name]:
                raise Rejected("missing_value", f"values.{name} is empty.", f"{param}.values.{name}")
            n *= len(q.values[name])
        if n > MAX_ALL_COMBINATIONS:
            raise Rejected("too_many_combinations", f'"all" gives {n:,} combinations; the limit is {MAX_ALL_COMBINATIONS:,}.', param)
        combos = [{}]
        for name in tpl.names:
            combos = [{**c, name: v} for c in combos for v in q.values[name]]
    elif q.combinations is None:
        if names:
            raise Rejected("missing_value", f"the template has placeholders {sorted(names)}: send `combinations`.", f"{param}.combinations")
        combos = [{}]
    else:
        if not q.combinations:
            raise Rejected("missing_value", "`combinations` is empty.", f"{param}.combinations")
        combos = q.combinations
        for i, c in enumerate(combos):
            if set(c) != names:
                raise _mismatch(set(c), names, f"{param}.combinations.{i}")
    for i, c in enumerate(combos):
        for name, v in c.items():
            if len(v) > MAX_VALUE_CHARS:
                raise Rejected("invalid_field", f"a value is over {MAX_VALUE_CHARS} characters.", f"{param}.combinations.{i}.{name}")
    return combos


def _mismatch(got: set[str], want: set[str], param: str) -> Rejected:
    if got - want:
        return Rejected("unknown_placeholder", f"{sorted(got - want)} not in the template.", param)
    return Rejected("missing_value", f"no value for {sorted(want - got)}.", param)


def check_options(q: Query, case_insensitive: bool, param: str, max_options: int = MAX_OPTIONS) -> None:
    if q.options is None:
        raise Rejected("missing_value", "a {?} query needs `options` (an extraction slot {*} {n} {$} needs none).", f"{param}.options")
    if not q.options:
        raise Rejected("too_many_options", "a query needs at least one option.", f"{param}.options")
    if len(q.options) > max_options:
        raise Rejected("too_many_options", f"{len(q.options):,} options; the limit is {max_options:,}.", f"{param}.options")
    seen: dict[str, int] = {}
    for i, o in enumerate(q.options):
        if not o:
            raise Rejected("invalid_field", "an option is empty.", f"{param}.options.{i}")
        k = o.strip().casefold() if case_insensitive else o
        if k in seen:
            raise Rejected(
                "duplicate_option",
                f"options {seen[k]} and {i} are the same{' ignoring case' if case_insensitive else ''}.",
                f"{param}.options.{i}",
            )
        seen[k] = i


def timestamp(t: float) -> str:
    """seconds -> "mm:ss" as Gemma's video prompt (transformers Gemma4Processor: int minutes, int seconds)"""
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def video_segments(v: Video, where: str, lim: Limits | None = None) -> list:
    """a video part -> [ "mm:ss ", Image, " mm:ss ", Image, ...] (Gemma4Processor.replace_video_token: frames joined by a space,
    each preceded by its timestamp and a space); frames in time order"""
    from spinf_serve import media

    lim = lim or Limits()
    if not v.frames:
        raise Rejected("invalid_field", "a video needs at least one frame.", f"{where}.video.frames")
    if len(v.frames) > lim.max_frames:
        raise Rejected("invalid_field", f"{len(v.frames):,} frames; at most {lim.max_frames:,} per video.", f"{where}.video.frames")
    out: list = []
    last = -1.0
    for k, f in enumerate(v.frames):
        if f.t < last:
            raise Rejected("invalid_field", "frames must be in time order (t not decreasing).", f"{where}.video.frames.{k}.t")
        last = f.t
        try:
            img = media.decode_image(f.image_url.url, v.detail, lim.max_image_pixels)
        except media.MediaError as e:
            raise Rejected("invalid_media", str(e), f"{where}.video.frames.{k}") from None
        img.inline = True
        out += [(" " if k else "") + timestamp(f.t) + " ", img]
    return out


MEDIA_PARTS = {"image_url": "image", "input_audio": "audio", "video": "video"}


def media_kinds(inp: Input) -> set[str]:
    """the media kinds (image, audio, video) of an input's parts, without decoding them: what a server gates"""
    return {
        MEDIA_PARTS[part.type]
        for msg in inp.messages
        if not isinstance(msg.content, str)
        for part in msg.content
        if part.type in MEDIA_PARTS
    }


def content_segments(inp: Input, param: str, lim: Limits | None = None) -> list:
    """an input's content as segments: str (text) | media.Image | media.Audio. Text parts of a message are joined as they are,
    messages by a blank line (a base model: roles are not rendered). Images go first, right after <bos>, in their order (as the
    model reads them best); audio stays where it is in the content, after the text that introduces it;
    a video's frames stay where the video is, each after its timestamp (Image.inline). The server's settings decide who may send media;
    here they are only decoded (invalid_media on bad data)."""
    from spinf_serve import media

    lim = lim or Limits()
    items: list = []
    for m, msg in enumerate(inp.messages):
        if m:
            items.append("\n\n")
        if isinstance(msg.content, str):
            items.append(msg.content)
            continue
        for k, part in enumerate(msg.content):
            where = f"{param}.messages.{m}.content.{k}"
            try:
                if part.type == "text" and part.text is not None:
                    items.append(part.text)
                elif part.type == "image_url" and part.image_url is not None:
                    items.append(media.decode_image(part.image_url.url, part.image_url.detail, lim.max_image_pixels))
                elif part.type == "input_audio" and part.input_audio is not None:
                    items.append(media.decode_audio(part.input_audio.data, part.input_audio.format, lim.max_audio_seconds))
                elif part.type == "video" and part.video is not None:
                    items += video_segments(part.video, where, lim)
                else:
                    raise Rejected("unsupported_content", f"content part type {part.type!r} is not supported.", where)
            except media.MediaError as e:
                raise Rejected("invalid_media", str(e), where) from None
    hoist = [x for x in items if isinstance(x, media.Image) and not x.inline]
    rest: list = []
    for x in items:
        if isinstance(x, media.Image) and not x.inline:
            continue
        if isinstance(x, str) and rest and isinstance(rest[-1], str):
            rest[-1] += x
        else:
            rest.append(x)
    return hoist + [x for x in rest if not (isinstance(x, str) and x == "")]


def content_text(inp: Input, param: str) -> str:
    """the text of an input's content (the text segments joined); for inputs without media, the whole content"""
    return "".join(x for x in content_segments(inp, param) if isinstance(x, str))
