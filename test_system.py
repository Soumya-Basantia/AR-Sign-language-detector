"""
test_system.py — Automated verification test suite for AR Sign Language system modules.
"""

import unittest
import numpy as np
from grammar_corrector import GrammarCorrector
from letter_buffer import LetterBuffer
from prediction_memory import PredictionMemory
from feature_extraction import extract_landmarks_from_results


class TestGrammarCorrector(unittest.TestCase):
    def setUp(self):
        self.corrector = GrammarCorrector()

    def test_noise_removal(self):
        res = self.corrector.correct(["UM", "NEED", "HELP"])
        self.assertNotIn("Um", res)
        self.assertIn("help", res.lower())

    def test_phrase_expansion(self):
        res = self.corrector.correct(["HELP", "PLEASE"])
        self.assertEqual(res, "Please help me!")

    def test_agreement(self):
        res = self.corrector.correct(["I", "IS", "HUNGRY"])
        self.assertEqual(res, "I am hungry.")

    def test_dedup(self):
        res = self.corrector.correct(["WATER", "WATER", "PLEASE"])
        self.assertEqual(res, "Please give me water.")

    def test_question_punctuation(self):
        res = self.corrector.correct(["WHERE", "DOCTOR"])
        self.assertTrue(res.endswith("?"))


class TestLetterBuffer(unittest.TestCase):
    def setUp(self):
        self.buf = LetterBuffer(pause_seconds=1.5)

    def test_add_and_flush(self):
        for ch in "HEPL":
            self.buf.add_letter(ch)
        flushed = self.buf.flush_on_space()
        self.assertEqual(flushed, ["HELP"])

    def test_spacing(self):
        for ch in "HI":
            self.buf.add_letter(ch)
        self.buf.add_space()
        for ch in "YU":
            self.buf.add_letter(ch)
        flushed = self.buf.flush_on_space()
        self.assertEqual(flushed, ["HI", "YOU"])


class TestPredictionMemory(unittest.TestCase):
    def setUp(self):
        self.memory = PredictionMemory(window=5, min_votes=3, stable_frames=3, confidence_threshold=0.5)

    def test_majority_voting(self):
        status = ""
        label = ""
        conf = 0.0
        for _ in range(5):
            self.memory.push("HELP", 0.9)
            label, conf, status = self.memory.get_stable_prediction()
        self.assertEqual(label, "HELP")
        self.assertEqual(status, "READY")
        self.assertGreaterEqual(conf, 0.8)


class TestFeatureExtraction(unittest.TestCase):
    def test_empty_results(self):
        feats, detected = extract_landmarks_from_results(None, include_face=False)
        self.assertEqual(feats.shape, (84,))
        self.assertFalse(detected)
        self.assertTrue(np.all(feats == 0))


if __name__ == "__main__":
    unittest.main()
