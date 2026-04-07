"""
GrammarCorrector — Lightweight rule-based sentence polishing.

No NLP models. Pure heuristics + lookup tables.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# Contraction / normalization table
# ---------------------------------------------------------------------------
NORMALISE = {
    "I AM": "I'M",
    "YOU ARE": "YOU'RE",
    "HE IS": "HE'S",
    "SHE IS": "SHE'S",
    "WE ARE": "WE'RE",
    "THEY ARE": "THEY'RE",
    "IT IS": "IT'S",
    "DO NOT": "DON'T",
    "CAN NOT": "CAN'T",
    "CANNOT": "CAN'T",
    "WILL NOT": "WON'T",
    "WANT TO": "WANT TO",
}

# Very small subject–verb agreement patches
AGREEMENT = {
    "I IS": "I AM",
    "YOU IS": "YOU ARE",
    "WE IS": "WE ARE",
    "THEY IS": "THEY ARE",
    "HE AM": "HE IS",
    "SHE AM": "SHE IS",
    "IT AM": "IT IS",
}

# Words that should always be capitalised
ALWAYS_CAPS = {"I"}

# Sentence-terminal punctuation triggers
SENTENCE_ENDERS = {"PLEASE", "NOW", "THANKS", "THANK", "YES", "NO", "BYE",
                   "STOP", "WAIT", "SORRY", "OKAY", "OK"}

# Filler / noise words to strip
NOISE_WORDS = {"UM", "UH", "HMM", "ERR"}


class GrammarCorrector:
    """
    Rule pipeline applied in order:
      1. Strip noise words
      2. Deduplicate consecutive identical words
      3. Subject-verb agreement patches
      4. Capitalise first word + 'I'
      5. Add terminal punctuation
      6. Optionally contract common phrases
    """

    def __init__(self, add_punctuation: bool = True,
                 apply_contractions: bool = False):
        self._add_punct = add_punctuation
        self._contract = apply_contractions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correct(self, words: List[str]) -> str:
        """
        Accept a list of words (already uppercase) and return a polished
        sentence string.
        """
        if not words:
            return ""

        tokens = [w.upper().strip() for w in words if w.strip()]

        tokens = self._strip_noise(tokens)
        tokens = self._dedup_consecutive(tokens)
        tokens = self._fix_agreement(tokens)
        tokens = self._capitalise(tokens)
        sentence = " ".join(tokens)

        if self._contract:
            sentence = self._apply_contractions(sentence)

        sentence = self._add_terminal_punct(tokens, sentence)
        return sentence

    def correct_string(self, sentence: str) -> str:
        """Convenience wrapper — accepts a space-separated string."""
        return self.correct(sentence.split())

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _strip_noise(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t not in NOISE_WORDS]

    def _dedup_consecutive(self, tokens: List[str]) -> List[str]:
        """Remove runs of identical words: HELP HELP HELP → HELP."""
        if not tokens:
            return tokens
        out = [tokens[0]]
        for t in tokens[1:]:
            if t != out[-1]:
                out.append(t)
        return out

    def _fix_agreement(self, tokens: List[str]) -> List[str]:
        """Patch obvious subject-verb mismatches."""
        out = list(tokens)
        for i in range(len(out) - 1):
            bigram = f"{out[i]} {out[i+1]}"
            if bigram in AGREEMENT:
                replacement = AGREEMENT[bigram].split()
                out[i], out[i + 1] = replacement[0], replacement[1]
        return out

    def _capitalise(self, tokens: List[str]) -> List[str]:
        """Title-case first word; keep 'I' always uppercase."""
        out = []
        for idx, t in enumerate(tokens):
            if idx == 0:
                out.append(t.capitalize())
            elif t in ALWAYS_CAPS:
                out.append(t)          # keep "I" uppercase
            else:
                out.append(t.capitalize())
        return out

    def _apply_contractions(self, sentence: str) -> str:
        upper = sentence.upper()
        for phrase, contraction in NORMALISE.items():
            if phrase in upper:
                # Case-insensitive replace preserving the contracted form
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                sentence = pattern.sub(contraction, sentence)
        return sentence

    def _add_terminal_punct(self, original_tokens: List[str],
                             sentence: str) -> str:
        if not sentence or sentence[-1] in ".!?,":
            return sentence
        if not self._add_punct:
            return sentence + "."

        last = original_tokens[-1].upper() if original_tokens else ""

        # Questions
        question_starters = {"WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW",
                              "CAN", "DO", "DOES", "IS", "ARE", "WILL"}
        first = original_tokens[0].upper() if original_tokens else ""
        if first in question_starters:
            return sentence + "?"

        # Exclamatory endings
        if last in {"HELP", "STOP", "NOW", "PLEASE"}:
            return sentence + "!"

        # Polite endings
        if last in SENTENCE_ENDERS - {"HELP", "STOP", "NOW", "PLEASE"}:
            return sentence + "."

        return sentence + "."
