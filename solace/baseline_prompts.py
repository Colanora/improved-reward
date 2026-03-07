import re
from typing import List, Optional


_QUOTED_TEXT_RE = re.compile(r'(["\'])(.*?)\1')
_COUNT_RE = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b", re.IGNORECASE)

_NUMBER_TO_WORD = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}
_WORD_TO_NUMBER = {word: number for number, word in _NUMBER_TO_WORD.items()}

_SPATIAL_RULES = [
    ("to the left of", "to the right of"),
    ("to the right of", "to the left of"),
    ("in front of", "behind"),
    ("behind", "in front of"),
    ("left of", "right of"),
    ("right of", "left of"),
    ("above", "below"),
    ("below", "above"),
    ("under", "over"),
    ("over", "under"),
]

_ATTRIBUTE_LEXICONS = {
    "color": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"],
    "size": ["small", "large", "tiny", "huge", "miniature", "giant"],
    "material": ["wooden", "metal", "glass", "stone", "plastic", "ceramic"],
}


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


def _replace_span(prompt: str, start: int, end: int, replacement: str) -> str:
    return prompt[:start] + replacement + prompt[end:]


def _rotate_alphanumeric(text: str, variant: int = 0) -> str:
    rotated = []
    digit_shift = 3 + variant
    upper_shift = 7 + variant
    lower_shift = 11 + variant

    for char in text:
        if "0" <= char <= "9":
            base = ord("0")
            rotated.append(chr(base + ((ord(char) - base + digit_shift) % 10)))
        elif "A" <= char <= "Z":
            base = ord("A")
            rotated.append(chr(base + ((ord(char) - base + upper_shift) % 26)))
        elif "a" <= char <= "z":
            base = ord("a")
            rotated.append(chr(base + ((ord(char) - base + lower_shift) % 26)))
        else:
            rotated.append(char)

    replacement = "".join(rotated)
    if replacement == text:
        return _rotate_alphanumeric(text, variant + 1)
    return replacement


def build_unconditional(prompt: str) -> str:
    del prompt
    return ""


def build_ocr_negative(prompt: str, variant: int = 0) -> Optional[str]:
    match = _QUOTED_TEXT_RE.search(prompt)
    if match is None:
        return None

    original = match.group(2)
    replacement = _rotate_alphanumeric(original, variant=variant)
    return _replace_span(prompt, match.start(2), match.end(2), replacement)


def build_count_negative(prompt: str, variant: int = 0) -> Optional[str]:
    match = _COUNT_RE.search(prompt)
    if match is None:
        return None

    token = match.group(1)
    lower = token.lower()
    value = int(token) if token.isdigit() else _WORD_TO_NUMBER.get(lower)
    if value is None:
        return None

    options = []
    if value < 12:
        options.append(value + 1)
    if value > 0:
        options.append(value - 1)
    if not options:
        return None

    target = options[variant % len(options)]
    replacement = str(target) if token.isdigit() else _preserve_case(token, _NUMBER_TO_WORD[target])
    return _replace_span(prompt, match.start(1), match.end(1), replacement)


def build_spatial_negative(prompt: str, variant: int = 0) -> Optional[str]:
    del variant
    lowered = prompt.lower()
    for source, target in _SPATIAL_RULES:
        match = re.search(rf"\b{re.escape(source)}\b", lowered)
        if match is None:
            continue
        original = prompt[match.start():match.end()]
        replacement = _preserve_case(original, target)
        return _replace_span(prompt, match.start(), match.end(), replacement)
    return None


def build_attribute_negative(prompt: str, variant: int = 0) -> Optional[str]:
    lowered = prompt.lower()
    for lexicon in _ATTRIBUTE_LEXICONS.values():
        for token in lexicon:
            match = re.search(rf"\b{re.escape(token)}\b", lowered)
            if match is None:
                continue
            alternatives = [candidate for candidate in lexicon if candidate != token]
            replacement_token = alternatives[variant % len(alternatives)]
            original = prompt[match.start():match.end()]
            replacement = _preserve_case(original, replacement_token)
            return _replace_span(prompt, match.start(), match.end(), replacement)
    return None


def resolve_counterfactual_mode(prompt: str, mode: str = "auto") -> str:
    if mode != "auto":
        if mode in {"unconditional", "pmi"}:
            return "unconditional"
        builder = {
            "ocr": build_ocr_negative,
            "count": build_count_negative,
            "spatial": build_spatial_negative,
            "attribute": build_attribute_negative,
        }.get(mode)
        if builder is None:
            raise ValueError(f"Unsupported counterfactual mode: {mode}")
        return mode if builder(prompt) is not None else "unconditional"

    ordered_builders = [
        ("ocr", build_ocr_negative),
        ("count", build_count_negative),
        ("spatial", build_spatial_negative),
        ("attribute", build_attribute_negative),
    ]
    for name, builder in ordered_builders:
        if builder(prompt) is not None:
            return name
    return "unconditional"


def build_counterfactuals(prompt: str, mode: str = "auto", n_neg: int = 1) -> List[str]:
    target_mode = resolve_counterfactual_mode(prompt, mode=mode)
    if target_mode == "unconditional":
        return [build_unconditional(prompt)]

    builder = {
        "ocr": build_ocr_negative,
        "count": build_count_negative,
        "spatial": build_spatial_negative,
        "attribute": build_attribute_negative,
    }[target_mode]

    negatives: List[str] = []
    max_trials = max(n_neg * 4, 4)
    for variant in range(max_trials):
        negative = builder(prompt, variant=variant)
        if negative is None or negative == prompt or negative in negatives:
            continue
        negatives.append(negative)
        if len(negatives) >= n_neg:
            break

    return negatives if negatives else [build_unconditional(prompt)]
