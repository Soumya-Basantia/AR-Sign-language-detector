"""
ai_expander.py — AI-Powered Sign Language to Fluent Speech Expansion
====================================================================
Integrates Google Gemini API to translate telegraphic sign language
glosses (e.g. ["COLD", "OUTSIDE", "NEED", "COAT"]) into natural,
expressive conversational sentences with tone adaptation.

Includes offline rule-based fallback using GrammarCorrector.
"""

import os
import json
import requests
from typing import List, Optional
from grammar_corrector import GrammarCorrector


class AIExpander:
    """
    Expands sign language keywords into fluent natural language sentences.
    Supports tone selection and graceful fallback to rule-based engine.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    FALLBACK_MODEL = "gemini-1.5-flash"

    TONE_PROMPTS = {
        "casual": "in a casual, natural conversational everyday tone",
        "polite": "in a polite, courteous, and respectful tone",
        "emergency": "in an urgent, clear emergency tone prioritizing immediate attention",
    }

    def __init__(self, api_key: Optional[str] = None, default_tone: str = "polite"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.default_tone = default_tone
        self.rule_corrector = GrammarCorrector(add_punctuation=True)
        self._cache = {}

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def expand(self, words: List[str], tone: Optional[str] = None) -> str:
        """
        Expand list of words into a full natural sentence.

        Args:
            words: List of uppercase signed tokens (e.g. ["NEED", "WATER", "PLEASE"])
            tone: "casual", "polite", or "emergency"

        Returns:
            Polished natural language string.
        """
        if not words:
            return ""

        # Filter empty
        clean_words = [w.strip().upper() for w in words if w.strip()]
        if not clean_words:
            return ""

        selected_tone = tone or self.default_tone
        cache_key = (tuple(clean_words), selected_tone)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If Gemini API key is available, use LLM expansion
        if self.api_key:
            llm_result = self._expand_with_gemini(clean_words, selected_tone)
            if llm_result:
                self._cache[cache_key] = llm_result
                return llm_result

        # Fallback to enhanced rule-based grammar corrector
        fallback_result = self.rule_corrector.correct(clean_words)
        self._cache[cache_key] = fallback_result
        return fallback_result

    def _expand_with_gemini(self, words: List[str], tone: str) -> Optional[str]:
        """Call Gemini API via REST generateContent endpoint."""
        tone_desc = self.TONE_PROMPTS.get(tone, self.TONE_PROMPTS["polite"])
        sign_string = " ".join(words)

        prompt = (
            f"You are an assistive communication bridge for a Deaf person signing in American Sign Language (ASL). "
            f"Convert the following signed sign glosses/keywords into a single fluent, natural English spoken sentence "
            f"{tone_desc}. "
            f"Maintain first-person perspective. Do NOT add quotes, markdown, explanations, or multiple options. "
            f"Output only the final spoken sentence.\n\n"
            f"Signed Tokens: {sign_string}"
        )

        for model_name in [self.DEFAULT_MODEL, self.FALLBACK_MODEL]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 60,
                }
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=4.0)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "").strip()
                            # Clean surrounding quotes
                            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                                text = text[1:-1].strip()
                            if text:
                                return text
                elif resp.status_code in (400, 403, 404):
                    # Try fallback model or stop
                    continue
            except Exception as e:
                # Network error or timeout, will fall back
                break

        return None
