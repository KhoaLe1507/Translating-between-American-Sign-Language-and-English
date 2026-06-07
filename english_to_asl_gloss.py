from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Iterable


"""
Rule-based English -> ASL gloss converter.

This module intentionally avoids heavyweight NLP dependencies so it can run on
the Raspberry Pi/client side with the rest of this project. It follows beginner
ASL gloss rules:

  - Time/tense information is moved to the beginning when it is explicit.
  - Basic clauses keep Subject-Verb-Object order unless a WH word is moved.
  - English function words such as articles, BE verbs, DO auxiliaries, and many
    prepositions are removed when they do not carry useful ASL gloss meaning.
  - Yes/No questions keep statement-like order, repeat the subject at the end,
    and end with '?'.
  - WH questions move the WH sign to the end and end with '?'.
  - The common phrase "How are you?" is kept as "HOW YOU?".
  - The common response "I'm fine" is kept as "ME FINE ME.".

This is a practical gloss normalizer, not a full ASL translator. Real ASL also
uses facial grammar, spatial agreement, classifiers, role shift, and context.
"""


@dataclass(frozen=True)
class ConversionResult:
    english: str
    gloss: str
    sentence_type: str


CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.I), replacement)
    for pattern, replacement in [
        (r"\bwon't\b", "will not"),
        (r"\bcan't\b", "can not"),
        (r"\bcannot\b", "can not"),
        (r"\bdon't\b", "do not"),
        (r"\bdoesn't\b", "does not"),
        (r"\bdidn't\b", "did not"),
        (r"\bisn't\b", "is not"),
        (r"\baren't\b", "are not"),
        (r"\bwasn't\b", "was not"),
        (r"\bweren't\b", "were not"),
        (r"\bhaven't\b", "have not"),
        (r"\bhasn't\b", "has not"),
        (r"\bhadn't\b", "had not"),
        (r"\bI'm\b", "I am"),
        (r"\byou're\b", "you are"),
        (r"\bhe's\b", "he is"),
        (r"\bshe's\b", "she is"),
        (r"\bit's\b", "it is"),
        (r"\bwe're\b", "we are"),
        (r"\bthey're\b", "they are"),
        (r"\bwhat's\b", "what is"),
        (r"\bwhere's\b", "where is"),
        (r"\bwhen's\b", "when is"),
        (r"\bwho's\b", "who is"),
        (r"\bwhy's\b", "why is"),
        (r"\bhow's\b", "how is"),
        (r"\bI'll\b", "I will"),
        (r"\byou'll\b", "you will"),
        (r"\bhe'll\b", "he will"),
        (r"\bshe'll\b", "she will"),
        (r"\bwe'll\b", "we will"),
        (r"\bthey'll\b", "they will"),
        (r"\bI've\b", "I have"),
        (r"\byou've\b", "you have"),
        (r"\bwe've\b", "we have"),
        (r"\bthey've\b", "they have"),
        (r"\bI'd\b", "I would"),
        (r"\byou'd\b", "you would"),
        (r"\bhe'd\b", "he would"),
        (r"\bshe'd\b", "she would"),
        (r"\bwe'd\b", "we would"),
        (r"\bthey'd\b", "they would"),
        (r"\bLet's\b", "let us"),
    ]
)


WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?::\d+)?(?:am|pm)?|[.!?]", re.I)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")

BE_FORMS = {"am", "m", "is", "are", "was", "were", "be", "been", "being"}
DO_FORMS = {"do", "does", "did"}
HAVE_FORMS = {"have", "has", "had"}
ARTICLE_WORDS = {"a", "an", "the"}

DROP_WORDS = {
    *ARTICLE_WORDS,
    *BE_FORMS,
    *DO_FORMS,
    "about",
    "as",
    "at",
    "in",
    "into",
    "of",
    "on",
    "onto",
    "than",
    "that",
    "to",
}

SEMANTIC_MODALS = {
    "can": "CAN",
    "could": "CAN",
    "may": "MAYBE",
    "might": "MAYBE",
    "must": "MUST",
    "should": "SHOULD",
    "would": "WOULD",
}

FUTURE_AUX = {"will", "shall"}
NEGATION_WORDS = {"not", "never", "no"}

WH_PHRASES = {
    ("how", "many"): "HOW-MANY",
    ("how", "much"): "HOW-MUCH",
    ("how", "old"): "HOW-OLD",
    ("what", "time"): "WHEN",
}

WH_WORDS = {
    "who": "WHO",
    "whom": "WHO",
    "whose": "WHOSE",
    "what": "WHAT",
    "where": "WHERE",
    "when": "WHEN",
    "why": "WHY",
    "which": "WHICH",
    "how": "HOW",
}

QUESTION_STARTERS = BE_FORMS | DO_FORMS | HAVE_FORMS | FUTURE_AUX | set(SEMANTIC_MODALS)

PRONOUN_GLOSS = {
    "i": "ME",
    "me": "ME",
    "my": "MY",
    "mine": "MINE",
    "you": "YOU",
    "your": "YOUR",
    "yours": "YOURS",
    "he": "HE",
    "him": "HIM",
    "his": "HIS",
    "she": "SHE",
    "her": "HER",
    "hers": "HERS",
    "it": "IT",
    "its": "ITS",
    "we": "WE",
    "us": "US",
    "our": "OUR",
    "ours": "OURS",
    "they": "THEY",
    "them": "THEM",
    "their": "THEIR",
    "theirs": "THEIRS",
}

DAYS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

TIME_ANCHORS = {
    *DAYS,
    *MONTHS,
    "today",
    "tomorrow",
    "yesterday",
    "tonight",
    "now",
    "later",
    "soon",
    "morning",
    "afternoon",
    "evening",
    "night",
    "noon",
    "midnight",
    "weekend",
    "weekday",
    "week",
    "month",
    "year",
    "day",
    "hour",
    "minute",
    "spring",
    "summer",
    "fall",
    "autumn",
    "winter",
    "breakfast",
    "lunch",
    "dinner",
}

TIME_MODIFIERS = {
    "last",
    "next",
    "this",
    "every",
    "each",
    "all",
    "early",
    "late",
}

TIME_PREPOSITIONS = {"on", "in", "at", "during", "before", "after", "until", "by"}
TIME_CONNECTORS = {"and", "or"}
AM_PM = {"am", "pm"}

IRREGULAR_GLOSS = {
    "ate": "EAT",
    "eaten": "EAT",
    "eats": "EAT",
    "became": "BECOME",
    "began": "BEGIN",
    "begun": "BEGIN",
    "bought": "BUY",
    "brought": "BRING",
    "came": "COME",
    "comes": "COME",
    "did": "DO",
    "does": "DO",
    "done": "FINISH",
    "drank": "DRINK",
    "drunk": "DRINK",
    "drove": "DRIVE",
    "driven": "DRIVE",
    "felt": "FEEL",
    "found": "FIND",
    "gave": "GIVE",
    "given": "GIVE",
    "goes": "GO",
    "going": "GO",
    "gone": "GO",
    "got": "GET",
    "gotten": "GET",
    "had": "HAVE",
    "has": "HAVE",
    "heard": "HEAR",
    "knew": "KNOW",
    "known": "KNOW",
    "left": "LEAVE",
    "liked": "LIKE",
    "lives": "LIVE",
    "made": "MAKE",
    "met": "MEET",
    "paid": "PAY",
    "ran": "RUN",
    "read": "READ",
    "rode": "RIDE",
    "said": "SAY",
    "saw": "SEE",
    "seen": "SEE",
    "sent": "SEND",
    "slept": "SLEEP",
    "spoke": "SPEAK",
    "spoken": "SPEAK",
    "studied": "STUDY",
    "studies": "STUDY",
    "swam": "SWIM",
    "taken": "TAKE",
    "taught": "TEACH",
    "thought": "THINK",
    "told": "TELL",
    "took": "TAKE",
    "understood": "UNDERSTAND",
    "wanted": "WANT",
    "went": "GO",
    "woke": "WAKE",
    "woken": "WAKE",
    "wrote": "WRITE",
    "written": "WRITE",
}

PAST_FORMS = set(IRREGULAR_GLOSS) - {"does", "goes", "going", "has", "lives", "studies", "eats"}


def convert_english_to_asl(text: str) -> str:
    """Convert English text into an uppercase ASL-style gloss string."""

    return " ".join(result.gloss for result in convert_text(text))


def convert_text(text: str) -> list[ConversionResult]:
    """Convert one or more English sentences and keep sentence metadata."""

    results: list[ConversionResult] = []
    for raw_sentence in split_sentences(text):
        tokens, terminal = tokenize(raw_sentence)
        if not tokens:
            continue
        results.append(convert_sentence(raw_sentence.strip(), tokens, terminal))
    return results


def convert_sentence(raw_sentence: str, tokens: list[str], terminal: str) -> ConversionResult:
    if is_how_are_you(tokens):
        return ConversionResult(raw_sentence, "HOW YOU?", "wh_question")
    if is_i_am_fine(tokens):
        return ConversionResult(raw_sentence, "ME FINE ME.", "statement")

    sentence_type = classify_sentence(tokens, terminal)
    wh_gloss, tokens_without_wh = extract_wh(tokens) if sentence_type == "wh_question" else (None, tokens)

    time_tokens, core_tokens = extract_time_phrase(tokens_without_wh)
    core_tokens, leading_markers, tail_markers = handle_auxiliary_order(core_tokens, sentence_type, bool(time_tokens))
    yes_no_subject = extract_yes_no_subject(core_tokens) if sentence_type == "yes_no_question" else []
    core_tokens, negation_markers = move_negation_to_end(core_tokens)

    core_gloss = normalize_core_tokens(core_tokens)
    time_gloss = normalize_core_tokens(time_tokens, drop_function_words=False)

    gloss_tokens = []
    gloss_tokens.extend(time_gloss)
    gloss_tokens.extend(leading_markers)
    gloss_tokens.extend(core_gloss)
    gloss_tokens.extend(tail_markers)
    gloss_tokens.extend(negation_markers)
    if wh_gloss:
        gloss_tokens.append(wh_gloss)

    gloss_tokens = dedupe_adjacent(gloss_tokens)
    gloss_tokens.extend(yes_no_subject)
    punctuation = "?" if sentence_type in {"yes_no_question", "wh_question"} else "."
    gloss = " ".join(gloss_tokens).strip()
    if gloss:
        gloss = f"{gloss}{punctuation}"
    return ConversionResult(raw_sentence, gloss, sentence_type)


def split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]


def tokenize(sentence: str) -> tuple[list[str], str]:
    normalized = normalize_text(sentence)
    pieces = WORD_RE.findall(normalized)
    terminal = "."
    if pieces and pieces[-1] in ".!?":
        terminal = "?" if pieces[-1] == "?" else "."
        pieces = pieces[:-1]
    return pieces, terminal


def normalize_text(text: str) -> str:
    normalized = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    for pattern, replacement in CONTRACTIONS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def classify_sentence(tokens: list[str], terminal: str) -> str:
    lowered = [token.lower() for token in tokens]
    has_wh = find_wh(lowered) is not None
    if has_wh and (terminal == "?" or lowered[0] in WH_WORDS):
        return "wh_question"
    if terminal == "?" or (lowered and lowered[0] in QUESTION_STARTERS):
        return "yes_no_question"
    return "statement"


def is_how_are_you(tokens: list[str]) -> bool:
    return [token.lower() for token in tokens] == ["how", "are", "you"]


def is_i_am_fine(tokens: list[str]) -> bool:
    return [token.lower() for token in tokens] in (["i", "am", "fine"], ["i", "m", "fine"])


def extract_wh(tokens: list[str]) -> tuple[str | None, list[str]]:
    lowered = [token.lower() for token in tokens]
    found = find_wh(lowered)
    if found is None:
        return None, tokens

    index, length, gloss = found
    without_wh = [token for pos, token in enumerate(tokens) if not index <= pos < index + length]
    return gloss, without_wh


def find_wh(lowered_tokens: list[str]) -> tuple[int, int, str] | None:
    for index in range(len(lowered_tokens)):
        for phrase, gloss in WH_PHRASES.items():
            if tuple(lowered_tokens[index : index + len(phrase)]) == phrase:
                return index, len(phrase), gloss
        word = lowered_tokens[index]
        if word in WH_WORDS:
            return index, 1, WH_WORDS[word]
    return None


def extract_time_phrase(tokens: list[str]) -> tuple[list[str], list[str]]:
    lowered = [token.lower() for token in tokens]
    used = [False] * len(tokens)
    time_tokens: list[str] = []
    index = 0

    while index < len(tokens):
        word = lowered[index]
        if word in TIME_PREPOSITIONS:
            probe = skip_articles(lowered, index + 1)
            if starts_time_phrase(lowered, probe):
                phrase_end, phrase_tokens, phrase_used = collect_time_phrase(tokens, lowered, probe)
                used[index] = True
                for pos in phrase_used:
                    used[pos] = True
                time_tokens.extend(phrase_tokens)
                index = phrase_end
                continue

        if starts_time_phrase(lowered, index):
            phrase_end, phrase_tokens, phrase_used = collect_time_phrase(tokens, lowered, index)
            for pos in phrase_used:
                used[pos] = True
            time_tokens.extend(phrase_tokens)
            index = phrase_end
            continue

        index += 1

    core_tokens = [token for pos, token in enumerate(tokens) if not used[pos]]
    return time_tokens, core_tokens


def skip_articles(lowered: list[str], index: int) -> int:
    while index < len(lowered) and lowered[index] in ARTICLE_WORDS:
        index += 1
    return index


def starts_time_phrase(lowered: list[str], index: int) -> bool:
    if index >= len(lowered):
        return False
    word = lowered[index]
    next_word = lowered[index + 1] if index + 1 < len(lowered) else ""
    if is_time_word(word):
        return True
    if word in TIME_MODIFIERS and (is_time_word(next_word) or is_clock_or_year(next_word)):
        return True
    if is_clock_or_year(word):
        return True
    return False


def collect_time_phrase(
    tokens: list[str], lowered: list[str], start_index: int
) -> tuple[int, list[str], set[int]]:
    phrase_tokens: list[str] = []
    used_positions: set[int] = set()
    index = start_index

    while index < len(tokens):
        word = lowered[index]
        next_word = lowered[index + 1] if index + 1 < len(lowered) else ""

        if word in ARTICLE_WORDS:
            used_positions.add(index)
            index += 1
            continue
        if word in TIME_PREPOSITIONS and starts_time_phrase(lowered, skip_articles(lowered, index + 1)):
            used_positions.add(index)
            index += 1
            continue
        if word in TIME_CONNECTORS and starts_time_phrase(lowered, skip_articles(lowered, index + 1)):
            phrase_tokens.append(tokens[index])
            used_positions.add(index)
            index += 1
            continue
        if word in TIME_MODIFIERS and (is_time_word(next_word) or is_clock_or_year(next_word)):
            phrase_tokens.append(tokens[index])
            used_positions.add(index)
            index += 1
            continue
        if word in AM_PM and phrase_tokens and is_clock_or_year(phrase_tokens[-1].lower()):
            phrase_tokens.append(tokens[index])
            used_positions.add(index)
            index += 1
            continue
        if is_time_word(word) or is_clock_or_year(word):
            phrase_tokens.append(tokens[index])
            used_positions.add(index)
            index += 1
            continue
        break

    return index, phrase_tokens, used_positions


def is_time_word(word: str) -> bool:
    return word in TIME_ANCHORS


def is_clock_or_year(word: str) -> bool:
    if not word:
        return False
    if re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", word, re.I):
        return True
    return bool(re.fullmatch(r"\d{4}", word))


def handle_auxiliary_order(
    tokens: list[str], sentence_type: str, has_explicit_time: bool
) -> tuple[list[str], list[str], list[str]]:
    lowered = [token.lower() for token in tokens]
    leading_markers: list[str] = []
    tail_markers: list[str] = []

    if lowered and lowered[0] in QUESTION_STARTERS:
        starter = lowered[0]
        tokens = tokens[1:]
        lowered = lowered[1:]
        if starter in FUTURE_AUX and not has_explicit_time:
            leading_markers.append("FUTURE")
        elif starter in SEMANTIC_MODALS:
            tail_markers.append(SEMANTIC_MODALS[starter])
        elif starter in HAVE_FORMS and sentence_type == "yes_no_question":
            tail_markers.append("FINISH")

    lowered = [token.lower() for token in tokens]
    if not has_explicit_time and any(word in FUTURE_AUX for word in lowered):
        leading_markers.append("FUTURE")
    elif sentence_type == "statement" and not has_explicit_time and has_past_signal(lowered):
        leading_markers.append("PAST")

    return tokens, leading_markers, tail_markers


def has_past_signal(lowered_tokens: Iterable[str]) -> bool:
    for word in lowered_tokens:
        if word in {"was", "were"} or word in PAST_FORMS:
            return True
        if len(word) > 4 and word.endswith("ed") and word not in {"need", "red"}:
            return True
    return False


def move_negation_to_end(tokens: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    negation_markers: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in NEGATION_WORDS:
            marker = "NEVER" if lowered == "never" else "NOT"
            negation_markers.append(marker)
        else:
            kept.append(token)
    return kept, dedupe_adjacent(negation_markers)


def extract_yes_no_subject(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and tokens[index].lower() in ARTICLE_WORDS | NEGATION_WORDS:
        index += 1
    if index >= len(tokens):
        return []

    first = tokens[index].lower()
    if first in {"my", "your", "his", "her", "our", "their"}:
        subject = [normalize_gloss_word(first)]
        next_index = index + 1
        while next_index < len(tokens) and tokens[next_index].lower() in ARTICLE_WORDS:
            next_index += 1
        if next_index < len(tokens):
            subject.append(normalize_gloss_word(tokens[next_index].lower()))
        return [token for token in subject if token]

    if first in PRONOUN_GLOSS:
        return [PRONOUN_GLOSS[first]]

    return [normalize_gloss_word(first)]


def normalize_core_tokens(tokens: list[str], *, drop_function_words: bool = True) -> list[str]:
    gloss_tokens: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in FUTURE_AUX:
            continue
        if drop_function_words and lowered in DROP_WORDS:
            continue
        if drop_function_words and lowered in HAVE_FORMS and looks_like_auxiliary_have(lowered, tokens):
            continue
        gloss = normalize_gloss_word(lowered)
        if gloss:
            gloss_tokens.append(gloss)
    return dedupe_adjacent(gloss_tokens)


def looks_like_auxiliary_have(word: str, tokens: list[str]) -> bool:
    lowered = [token.lower() for token in tokens]
    try:
        index = lowered.index(word)
    except ValueError:
        return False
    if index + 1 >= len(lowered):
        return False
    next_word = lowered[index + 1]
    return next_word in PAST_FORMS or next_word.endswith("ed")


def normalize_gloss_word(word: str) -> str:
    if not word:
        return ""
    if word in PRONOUN_GLOSS:
        return PRONOUN_GLOSS[word]
    if re.fullmatch(r"\d+(?::\d+)?(am|pm)?", word, re.I):
        return word.upper()
    if word in IRREGULAR_GLOSS:
        return IRREGULAR_GLOSS[word]
    stem = simple_stem(word)
    return stem.upper()


def simple_stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith("mak"):
            return base + "e"
        return base
    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(word) > 4 and word.endswith("es") and not word.endswith(("ses", "ies")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def dedupe_adjacent(tokens: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for token in tokens:
        if token and (not deduped or deduped[-1] != token):
            deduped.append(token)
    return deduped


EXAMPLES: tuple[tuple[str, str], ...] = (
    ("I go to school on Saturday.", "SATURDAY ME GO SCHOOL."),
    ("I am happy today.", "TODAY ME HAPPY."),
    ("The boy went to the park yesterday.", "YESTERDAY BOY GO PARK."),
    ("Do you want water?", "YOU WANT WATER YOU?"),
    ("Are you a student?", "YOU STUDENT YOU?"),
    ("Can you help me?", "YOU HELP ME CAN YOU?"),
    ("Where do you live?", "YOU LIVE WHERE?"),
    ("How are you?", "HOW YOU?"),
    ("I'm fine.", "ME FINE ME."),
    ("I ' m fine.", "ME FINE ME."),
    ("I ' m happy.", "ME HAPPY."),
    ("What is your name?", "YOUR NAME WHAT?"),
    ("Why are you late?", "YOU LATE WHY?"),
    ("I will meet my friend tomorrow.", "TOMORROW ME MEET MY FRIEND."),
    ("I do not like coffee.", "ME LIKE COFFEE NOT."),
)


def run_examples() -> None:
    print("English -> ASL gloss examples")
    for english, expected in EXAMPLES:
        actual = convert_english_to_asl(english)
        status = "OK" if actual == expected else "CHECK"
        print(f"[{status}] {english} -> {actual}")
        if actual != expected:
            print(f"       expected: {expected}")


def assert_examples() -> None:
    failures = [
        (english, expected, convert_english_to_asl(english))
        for english, expected in EXAMPLES
        if convert_english_to_asl(english) != expected
    ]
    if failures:
        raise SystemExit(1)


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("English to ASL Gloss")
    root.geometry("760x520")
    root.minsize(620, 420)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = "#f5f7f6"
    text_bg = "#ffffff"
    accent = "#176b87"
    root.configure(bg=bg)

    style.configure("App.TFrame", background=bg)
    style.configure("Title.TLabel", background=bg, foreground="#172126", font=("Segoe UI", 18, "bold"))
    style.configure("Field.TLabel", background=bg, foreground="#334148", font=("Segoe UI", 10, "bold"))
    style.configure("Status.TLabel", background=bg, foreground="#5e6a70", font=("Segoe UI", 9))
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    style.map("Accent.TButton", foreground=[("active", accent)])

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    main = ttk.Frame(root, padding=16, style="App.TFrame")
    main.grid(row=0, column=0, sticky="nsew")
    main.columnconfigure(0, weight=1)
    main.rowconfigure(2, weight=1)
    main.rowconfigure(5, weight=1)

    title = ttk.Label(main, text="English to ASL Gloss", style="Title.TLabel")
    title.grid(row=0, column=0, sticky="w", pady=(0, 14))

    input_label = ttk.Label(main, text="English input", style="Field.TLabel")
    input_label.grid(row=1, column=0, sticky="w", pady=(0, 6))

    input_frame = ttk.Frame(main)
    input_frame.grid(row=2, column=0, sticky="nsew")
    input_frame.rowconfigure(0, weight=1)
    input_frame.columnconfigure(0, weight=1)

    input_text = tk.Text(
        input_frame,
        height=8,
        wrap="word",
        undo=True,
        font=("Segoe UI", 12),
        bg=text_bg,
        fg="#11181c",
        insertbackground="#11181c",
        relief="solid",
        borderwidth=1,
        padx=10,
        pady=8,
    )
    input_scroll = ttk.Scrollbar(input_frame, orient="vertical", command=input_text.yview)
    input_text.configure(yscrollcommand=input_scroll.set)
    input_text.grid(row=0, column=0, sticky="nsew")
    input_scroll.grid(row=0, column=1, sticky="ns")

    button_frame = ttk.Frame(main, style="App.TFrame")
    button_frame.grid(row=3, column=0, sticky="ew", pady=12)
    button_frame.columnconfigure(4, weight=1)

    output_label = ttk.Label(main, text="ASL gloss output", style="Field.TLabel")
    output_label.grid(row=4, column=0, sticky="w", pady=(0, 6))

    output_frame = ttk.Frame(main)
    output_frame.grid(row=5, column=0, sticky="nsew")
    output_frame.rowconfigure(0, weight=1)
    output_frame.columnconfigure(0, weight=1)

    output_text = tk.Text(
        output_frame,
        height=8,
        wrap="word",
        font=("Consolas", 14, "bold"),
        bg=text_bg,
        fg="#0f3f4f",
        relief="solid",
        borderwidth=1,
        padx=10,
        pady=8,
        state="disabled",
    )
    output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=output_text.yview)
    output_text.configure(yscrollcommand=output_scroll.set)
    output_text.grid(row=0, column=0, sticky="nsew")
    output_scroll.grid(row=0, column=1, sticky="ns")

    status_var = tk.StringVar(value="Ready")
    status = ttk.Label(main, textvariable=status_var, style="Status.TLabel")
    status.grid(row=6, column=0, sticky="w", pady=(10, 0))

    example_index = {"value": 0}

    def set_output(value: str) -> None:
        output_text.configure(state="normal")
        output_text.delete("1.0", "end")
        output_text.insert("1.0", value)
        output_text.configure(state="disabled")

    def convert_current_text(event: tk.Event | None = None) -> str:
        english = input_text.get("1.0", "end").strip()
        if not english:
            set_output("")
            status_var.set("Ready")
            return "break"

        gloss = convert_english_to_asl(english)
        set_output(gloss)
        status_var.set("Converted")
        return "break"

    def clear_text() -> None:
        input_text.delete("1.0", "end")
        set_output("")
        status_var.set("Cleared")
        input_text.focus_set()

    def load_example() -> None:
        english, _expected = EXAMPLES[example_index["value"]]
        example_index["value"] = (example_index["value"] + 1) % len(EXAMPLES)
        input_text.delete("1.0", "end")
        input_text.insert("1.0", english)
        convert_current_text()
        input_text.focus_set()

    def copy_output() -> None:
        gloss = output_text.get("1.0", "end").strip()
        if not gloss:
            status_var.set("Nothing to copy")
            return
        root.clipboard_clear()
        root.clipboard_append(gloss)
        status_var.set("Copied")

    convert_button = ttk.Button(
        button_frame,
        text="Convert",
        command=convert_current_text,
        style="Accent.TButton",
    )
    convert_button.grid(row=0, column=0, sticky="w", padx=(0, 8))

    ttk.Button(button_frame, text="Example", command=load_example).grid(row=0, column=1, sticky="w", padx=(0, 8))
    ttk.Button(button_frame, text="Clear", command=clear_text).grid(row=0, column=2, sticky="w", padx=(0, 8))
    ttk.Button(button_frame, text="Copy", command=copy_output).grid(row=0, column=3, sticky="w")

    root.bind("<Control-Return>", convert_current_text)
    input_text.focus_set()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert English text to ASL-style gloss text.")
    parser.add_argument("text", nargs="*", help="English text to convert. If omitted, the GUI is opened.")
    parser.add_argument("--examples", action="store_true", help="Print bundled conversion examples.")
    parser.add_argument("--self-test", action="store_true", help="Run bundled examples and assert expected output.")
    args = parser.parse_args()

    if args.self_test:
        run_examples()
        assert_examples()
        return

    if args.examples:
        run_examples()
        return

    if args.text:
        print(convert_english_to_asl(" ".join(args.text)))
        return

    launch_gui()


if __name__ == "__main__":
    main()
