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


class TestBaselineModel(unittest.TestCase):

    def test_baseline_models_exist_and_predict(self):
        from generate_baseline_model import ensure_baseline_models
        from predict_sequence import Predictor
        ensure_baseline_models()
        
        predictor = Predictor(model="word")
        self.assertFalse(predictor.demo_mode)
        # Test prediction with mock 40x84 array
        dummy_window = np.zeros((40, 84), dtype=np.float32)
        label, conf = predictor.predict(dummy_window)
        self.assertIsInstance(label, str)
        self.assertIsInstance(conf, float)


class TestAIExpander(unittest.TestCase):
    def setUp(self):
        from ai_expander import AIExpander
        self.expander = AIExpander(api_key="")  # tests rule-based fallback without API key

    def test_tones(self):
        words = ["HELP", "PLEASE"]
        res_polite = self.expander.expand(words, tone="polite")
        res_casual = self.expander.expand(words, tone="casual")
        res_emergency = self.expander.expand(words, tone="emergency")

        self.assertIn("help", res_polite.lower())
        self.assertIn("help", res_casual.lower())
        self.assertIn("help", res_emergency.lower())
        self.assertTrue(res_emergency.endswith("!"))

    def test_empty_input(self):
        res = self.expander.expand([], tone="casual")
        self.assertEqual(res, "")


class TestTwoWayBridge(unittest.TestCase):
    def setUp(self):
        from two_way_bridge import TwoWayBridge
        self.bridge = TwoWayBridge()

    def test_dialogue_flow(self):
        self.bridge.add_signer_message("I need some water, please.", raw_signs=["NEED", "WATER"])
        self.bridge.add_partner_message("Sure, let me get you a glass.")

        history = self.bridge.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["speaker"], "Signer")
        self.assertEqual(history[1]["speaker"], "Partner")

        subtitle = self.bridge.get_latest_partner_subtitle()
        self.assertIn("Sure, let me get you a glass.", subtitle)

        exported = self.bridge.export_json()
        self.assertIn("Sure, let me get you a glass.", exported)


class TestWebApp(unittest.TestCase):
    def test_api_endpoints(self):
        from fastapi.testclient import TestClient
        from web_app import app
        client = TestClient(app)

        res_status = client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertIn("models", res_status.json())

        res_vocab = client.get("/api/vocabulary")
        self.assertEqual(res_vocab.status_code, 200)
        self.assertIn("words", res_vocab.json())

        res_expand = client.post("/api/expand", json={"words": ["NEED", "HELP"], "tone": "emergency"})
        self.assertEqual(res_expand.status_code, 200)
        data = res_expand.json()
        self.assertIn("expanded_sentence", data)
        self.assertTrue(data["expanded_sentence"].endswith("!"))


        res_turn = client.post("/api/dialogue/partner", json={"text": "Hello, how can I help you today?"})
        self.assertEqual(res_turn.status_code, 200)
        self.assertEqual(res_turn.json()["status"], "ok")



if __name__ == "__main__":
    unittest.main()

