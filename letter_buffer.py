"""
LetterBuffer — Accumulates signed letters into words with auto-correction.
"""

import time
from collections import Counter


# Lightweight correction dictionary (common ASL spelling errors)
CORRECTION_MAP = {
    "HEPL": "HELP",
    "HLEP": "HELP",
    "WATR": "WATER",
    "WTER": "WATER",
    "PLEESE": "PLEASE",
    "PLESE": "PLEASE",
    "THNK": "THANK",
    "THANKU": "THANK YOU",
    "YU": "YOU",
    "WAN": "WANT",
    "WRTR": "WATER",
    "THNAK": "THANK",
    "MOER": "MORE",
    "MOR": "MORE",
    "PIAN": "PAIN",
    "HURTS": "HURT",
    "HLPO": "HELP",
    "MROE": "MORE",
    "NEDE": "NEED",
    "NED": "NEED",
    "OEPN": "OPEN",
    "OPNE": "OPEN",
    "CLOS": "CLOSE",
    "CLSOE": "CLOSE",
    "TIOLET": "TOILET",
    "TOLIET": "TOILET",
    "HUNGR": "HUNGRY",
    "HUNGERY": "HUNGRY",
    "TIERD": "TIRED",
    "TIREF": "TIRED",
}

# Common word dictionary for fuzzy matching
COMMON_WORDS = [
    "HELP", "WATER", "PLEASE", "THANK", "YOU", "WANT", "NEED", "MORE",
    "STOP", "YES", "NO", "OPEN", "CLOSE", "TOILET", "HUNGRY", "TIRED",
    "PAIN", "HURT", "CALL", "DOCTOR", "FOOD", "DRINK", "SLEEP", "HOME",
    "GOOD", "BAD", "HOT", "COLD", "HAPPY", "SAD", "LOVE", "SORRY",
    "HELLO", "BYE", "NAME", "WHERE", "WHEN", "WHAT", "WHO", "HOW",
    "COME", "GO", "WAIT", "NOW", "LATER", "TODAY", "TOMORROW",
]


def _levenshtein(a: str, b: str) -> int:
    """Fast Levenshtein distance."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def auto_correct(word: str) -> str:
    """Apply correction map then fuzzy dictionary matching."""
    if not word:
        return word
    upper = word.upper()

    # Direct map hit
    if upper in CORRECTION_MAP:
        return CORRECTION_MAP[upper]

    # Already a known word
    if upper in COMMON_WORDS:
        return upper

    # Fuzzy match — only correct if very close (distance ≤ 2) and word short enough
    if len(upper) >= 3:
        best_word, best_dist = None, 3  # threshold
        for w in COMMON_WORDS:
            d = _levenshtein(upper, w)
            if d < best_dist:
                best_dist, best_word = d, w
        if best_word:
            return best_word

    return upper


class LetterBuffer:
    """
    Accumulates letters signed one-by-one and forms corrected words.

    Usage
    -----
    buf = LetterBuffer(pause_seconds=1.5)
    buf.add_letter("H")
    buf.add_letter("E")
    buf.add_letter("L")
    buf.add_letter("P")
    word = buf.flush()          # → "HELP"
    word = buf.flush_on_space() # same, triggered by SPACE key
    """

    def __init__(self, pause_seconds: float = 1.5, max_letters: int = 20):
        self._letters: list[str] = []
        self._last_add_time: float = 0.0
        self._pause_seconds = pause_seconds
        self._max_letters = max_letters

        # Duplicate guard — don't add the same letter twice in a row
        self._last_letter: str = ""
        self._same_letter_cooldown: float = 0.8  # seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_letter(self, letter: str) -> bool:
        """
        Add a letter to the buffer.
        Returns True if letter was accepted, False if rejected (duplicate / cooldown).
        """
        letter = letter.upper()
        now = time.time()

        # Reject same-letter spam
        if (letter == self._last_letter and
                now - self._last_add_time < self._same_letter_cooldown):
            return False

        if len(self._letters) >= self._max_letters:
            return False

        self._letters.append(letter)
        self._last_letter = letter
        self._last_add_time = now
        return True

    def delete_last(self) -> str | None:
        """Remove and return the last letter."""
        if self._letters:
            removed = self._letters.pop()
            return removed
        return None

    def flush(self, correct: bool = True) -> str:
        """Return the accumulated word and clear the buffer."""
        word = "".join(self._letters)
        self._letters.clear()
        self._last_letter = ""
        if correct and word:
            word = auto_correct(word)
        return word

    def flush_on_space(self) -> str:
        """Alias for flush — call this when SPACE is pressed."""
        return self.flush(correct=True)

    def check_pause_flush(self) -> str | None:
        """
        Call every frame. Returns a flushed word if the user has paused
        long enough; otherwise returns None.
        """
        if not self._letters:
            return None
        if time.time() - self._last_add_time >= self._pause_seconds:
            return self.flush(correct=True)
        return None

    def reset(self):
        """Clear the buffer completely."""
        self._letters.clear()
        self._last_letter = ""
        self._last_add_time = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def letters(self) -> list[str]:
        return list(self._letters)

    @property
    def display_string(self) -> str:
        """Space-separated letters for AR display, e.g. 'H E L P'."""
        return " ".join(self._letters)

    @property
    def is_empty(self) -> bool:
        return len(self._letters) == 0

    @property
    def current_word_raw(self) -> str:
        return "".join(self._letters)
