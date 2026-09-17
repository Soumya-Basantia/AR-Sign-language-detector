"""
two_way_bridge.py — Bidirectional Assistive Communication Bridge
=================================================================
Manages multi-turn conversation between:
1. Signer (Sign Language user -> Gesture Recognition -> Speech Output)
2. Hearing Partner (Spoken English -> Speech-to-Text -> AR Glasses Subtitle)

Maintains full transcript history with export capabilities.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional


class TwoWayBridge:
    """
    Coordinates dialog between Signer and Hearing Partner.
    """

    def __init__(self, max_turns: int = 50):
        self._history: List[Dict[str, str]] = []
        self._max_turns = max_turns
        self._latest_partner_message: str = ""
        self._latest_partner_time: float = 0.0

    def add_signer_message(self, text: str, raw_signs: Optional[List[str]] = None, tone: str = "polite") -> Dict[str, str]:
        """Record an utterance produced by the sign language user."""
        turn = {
            "speaker": "Signer",
            "text": text.strip(),
            "raw_signs": " ".join(raw_signs) if raw_signs else "",
            "tone": tone,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "iso_time": datetime.now().isoformat()
        }
        self._append_turn(turn)
        return turn

    def add_partner_message(self, text: str) -> Dict[str, str]:
        """Record an utterance spoken by the hearing communication partner."""
        cleaned = text.strip()
        if not cleaned:
            return {}
        turn = {
            "speaker": "Partner",
            "text": cleaned,
            "tone": "spoken",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "iso_time": datetime.now().isoformat()
        }
        self._append_turn(turn)
        self._latest_partner_message = cleaned
        import time
        self._latest_partner_time = time.time()
        return turn

    def get_latest_partner_subtitle(self, timeout_seconds: float = 8.0) -> str:
        """
        Get latest speech from partner to float on the AR HUD.
        Returns empty string if timeout has elapsed.
        """
        import time
        if time.time() - self._latest_partner_time > timeout_seconds:
            return ""
        return self._latest_partner_message

    def get_history(self, limit: int = 25) -> List[Dict[str, str]]:
        return self._history[-limit:]

    def clear(self):
        self._history.clear()
        self._latest_partner_message = ""

    def export_text(self) -> str:
        """Export formatted text transcript."""
        lines = [
            f"=== AR Sign Communication Transcript ===",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 45
        ]
        for turn in self._history:
            spk = turn.get("speaker", "Unknown")
            txt = turn.get("text", "")
            t = turn.get("timestamp", "")
            lines.append(f"[{t}] {spk}: {txt}")
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export transcript as JSON string."""
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "turns_count": len(self._history),
            "transcript": self._history
        }, indent=2)

    def _append_turn(self, turn: Dict[str, str]):
        self._history.append(turn)
        if len(self._history) > self._max_turns:
            self._history.pop(0)
