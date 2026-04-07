"""
predict_sequence.py — Real-Time Assistive Communication System
==============================================================

Architecture
------------
Camera → MediaPipe Hands → Landmark extraction → Sliding window (40 frames)
    → Predictor → PredictionMemory (stability + context)
    → LetterBuffer (LETTER mode) / SentenceBuilder (WORD mode)
    → GrammarCorrector → ARDisplay (HUD overlay)

Key bindings
------------
  SPACE  → Accept current prediction into sentence / flush letter buffer
  D      → Delete last word / last letter
  R      → Full reset
  M      → Toggle WORD ↔ LETTER mode
  ENTER  → Finalize sentence (grammar correct + speak)
  G      → Toggle AR glasses mode ↔ Normal UI
  Q      → Quit

Run
---
  python predict_sequence.py --model models/your_model.pkl

If no model is available, the system launches in DEMO mode with
synthetic predictions so all UI features can be evaluated.
"""

import argparse
from pyexpat import features
import time
import joblib
import sys
import os
import threading
from collections import deque
from typing import Optional, List, Deque

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# Optional imports — graceful fallback
# ---------------------------------------------------------------------------
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[WARN] mediapipe not installed — running in DEMO mode.")

try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not installed — voice output disabled.")

try:
    import pickle
    _PICKLE_AVAILABLE = True
except ImportError:
    _PICKLE_AVAILABLE = False

# Local modules
from collect_sequences import IMPORTANT_FACE_POINTS
from letter_buffer import LetterBuffer
from grammar_corrector import GrammarCorrector
from prediction_memory import PredictionMemory
from ar_display import ARDisplay

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE        = 40       # frames per prediction window
FRAME_SKIP         = 2        # process every Nth frame
CONF_THRESHOLD     = 0.70
SENTENCE_MAX_WORDS = 10
FPS_SMOOTH_WINDOW  = 30

DEMO_WORDS  = ["HELP", "WATER", "PLEASE", "MORE", "THANK", "YOU",
               "NEED", "FOOD", "STOP", "DOCTOR"]
DEMO_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Sentence builder
# ---------------------------------------------------------------------------
class SentenceBuilder:
    """Rolling word buffer with duplicate guard."""

    def __init__(self, max_words: int = SENTENCE_MAX_WORDS,
                 cooldown: float = 1.2):
        self._words: List[str] = []
        self._max = max_words
        self._last_word: str = ""
        self._last_time: float = 0.0
        self._cooldown = cooldown

    def add(self, word: str) -> bool:
        """Add word; return True if accepted."""
        word = word.strip().upper()
        if not word:
            return False
        now = time.time()
        if word == self._last_word and now - self._last_time < self._cooldown:
            return False
        if len(self._words) >= self._max:
            self._words.pop(0)
        self._words.append(word)
        self._last_word = word
        self._last_time = now
        return True

    def delete_last(self) -> Optional[str]:
        if self._words:
            return self._words.pop()
        return None

    def reset(self):
        self._words.clear()
        self._last_word = ""

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def display(self) -> str:
        return " ".join(self._words)


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------
class Predictor:
    """
    Wraps a trained sklearn / custom model.
    Expects predict_proba(X) or predict(X).
    Falls back to DEMO synthetic predictions.
    """

    def __init__(self, model_path= "model/model_mlp.pkl"):

        self._demo = False

        # Load trained model
        self._model = joblib.load(model_path)
        self._classes = joblib.load("model/label_encoder.pkl").classes_
        
    def predict(self, window: np.ndarray):
    
        try:
            x = np.array(window.flatten(), dtype=np.float32).reshape(1, -1)
            x = np.nan_to_num(x)

            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(x)[0]
                idx = np.argmax(probs)
                conf = float(probs[idx])
                label = self._classes[idx]
            else:
                label = str(self._model.predict(x)[0])
                conf = 1.0

            print(f"[DEBUG] Prediction: {label}, Confidence: {conf:.2f}")
            return label, conf

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "---", 0.0

# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

def extract_landmarks(results, frame_rgb):
    if not results or not results.multi_hand_landmarks:
        return None

    left_hand = np.zeros(42)
    right_hand = np.zeros(42)

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):
        lm = np.array([[p.x, p.y, ] for p in hand_landmarks.landmark],
                      dtype=np.float32).flatten()
        label = handedness.classification[0].label

        if label == "Left":
            left_hand = lm
        else:
            right_hand = lm

    # ── Face ───────────────────────────────
    face_result = face_detector.process(frame_rgb)
    face_pts = np.zeros(len(IMPORTANT_FACE_POINTS) * 2)

    if face_result.multi_face_landmarks:
        lms = face_result.multi_face_landmarks[0].landmark
        coords = []
        for idx in IMPORTANT_FACE_POINTS:
            coords.extend([lms[idx].x, lms[idx].y])
        face_pts = np.array(coords)

    # Combine both hands and face → 124 features
    return np.concatenate([left_hand, right_hand, face_pts])

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
_tts_engine = None

def _init_tts():
    global _tts_engine
    if not _TTS_AVAILABLE:
        return
    try:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 150)
        _tts_engine.setProperty("volume", 0.9)
    except Exception as e:
        print(f"[WARN] TTS init failed: {e}")
        _tts_engine = None


def speak(text: str):
    """Speak text in a background thread (non-blocking)."""
    if _tts_engine is None:
        return
    def _run():
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

mp_face  = mp.solutions.face_mesh
face_detector = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(model_path: Optional[str] = None):
    _init_tts()

    # Components
    predictor   = Predictor(model_path)
    memory      = PredictionMemory(window=7, min_votes=4,
                                   stable_frames=5,
                                   confidence_threshold=CONF_THRESHOLD)
    letter_buf  = LetterBuffer(pause_seconds=1.5)
    sentence_b  = SentenceBuilder(max_words=SENTENCE_MAX_WORDS)
    corrector   = GrammarCorrector(add_punctuation=True)
    ar_display  = ARDisplay()

    # MediaPipe
    hands_proc = None
    if _MP_AVAILABLE:
        mp_hands  = mp.solutions.hands
        mp_draw   = mp.solutions.drawing_utils
        hands_proc = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    # State
    mode: str = "WORD"          # "WORD" or "LETTER"
    finalized: str = ""
    frame_idx: int = 0
    landmark_window: Deque = deque(maxlen=WINDOW_SIZE)

    # FPS counter
    fps_deque: Deque = deque(maxlen=FPS_SMOOTH_WINDOW)
    prev_time = time.time()

    # Current stable prediction
    cur_pred: str    = ""
    cur_conf: float  = 0.0
    cur_status: str  = ""

    hands_detected: bool = False

    print("\n" + "="*60)
    print("  SIGN LANGUAGE COMMUNICATION SYSTEM — READY")
    print("="*60)
    print("  Controls:")
    print("    SPACE  → Accept word/letter")
    print("    D      → Delete last")
    print("    R      → Reset")
    print("    M      → Toggle mode (WORD/LETTER)")
    print("    ENTER  → Finalize + speak")
    print("    G      → Toggle AR glasses mode")
    print("    Q      → Quit")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break
        frame = cv2.flip(frame, 1)
        frame_idx += 1

        # ── FPS ─────────────────────────────────────────────────────────
        now = time.time()
        fps_deque.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_deque))

        # ── Frame skip ─────────────────────────────────────────────────
        process_this_frame = (frame_idx % FRAME_SKIP == 0)

        if process_this_frame:
            if _MP_AVAILABLE and hands_proc:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_proc.process(rgb)
                lm = extract_landmarks(results, rgb)
                hands_detected = lm is not None

                # Draw skeleton
                if results and results.multi_hand_landmarks:
                    for hl in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hl,
                            mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0,200,150),
                                                thickness=2, circle_radius=3),
                            mp_draw.DrawingSpec(color=(0,150,200),
                                                thickness=2))
            else:
                lm = None
                hands_detected = False

            if lm is not None:
                landmark_window.append(lm)

            # Prediction when window full
            if len(landmark_window) == WINDOW_SIZE and hands_detected:
                window_arr = np.array(landmark_window)   # (40, 63)
                raw_label, raw_conf = predictor.predict(window_arr)
                memory.push(raw_label, raw_conf)

                stable_label, eff_conf, status = memory.get_stable_prediction()
                cur_pred   = stable_label or raw_label
                cur_conf   = eff_conf
                cur_status = status
            elif not hands_detected:
                cur_status = "NO HANDS"
                cur_pred   = ""
                cur_conf   = 0.0
            else:
                cur_status = "STABILIZING"

        # ── Pause-based letter flush ───────────────────────────────────
        if mode == "LETTER":
            flushed = letter_buf.check_pause_flush()
            if flushed:
                sentence_b.add(flushed)
                memory.confirm(flushed)
                print(f"[AUTO-FLUSH] '{flushed}'  →  {sentence_b.display}")

        # ── Key handling ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('g'):
            ar_display.toggle_mode()
            m = "AR GLASSES" if ar_display.ar_mode else "NORMAL"
            print(f"[UI] Switched to {m} mode")

        elif key == ord('m'):
            mode = "LETTER" if mode == "WORD" else "WORD"
            letter_buf.reset()
            print(f"[MODE] Switched to {mode}")

        elif key == ord('r'):
            sentence_b.reset()
            letter_buf.reset()
            memory.reset()
            finalized = ""
            print("[RESET] System reset.")

        elif key == ord('d'):
            if mode == "LETTER" and not letter_buf.is_empty:
                removed = letter_buf.delete_last()
                print(f"[DELETE] Removed letter '{removed}'")
            else:
                removed = sentence_b.delete_last()
                memory.delete_last_confirmed()
                print(f"[DELETE] Removed word '{removed}'")

        elif key == 32:  # SPACE
            if cur_status == "READY" and cur_conf >= CONF_THRESHOLD:
                if mode == "LETTER":
                    accepted = letter_buf.add_letter(cur_pred)
                    if accepted:
                        print(f"[LETTER] Added '{cur_pred}' → "
                              f"{letter_buf.display_string}")
                else:
                    accepted = sentence_b.add(cur_pred)
                    if accepted:
                        memory.confirm(cur_pred)
                        print(f"[WORD] Added '{cur_pred}' → "
                              f"{sentence_b.display}")
            else:
                print(f"[SKIP] status={cur_status} conf={cur_conf:.2f}")

        elif key == 13:  # ENTER
            if mode == "LETTER" and not letter_buf.is_empty:
                # Flush remaining letters as a word first
                flushed = letter_buf.flush_on_space()
                if flushed:
                    sentence_b.add(flushed)

            raw_words = sentence_b.words
            if raw_words:
                finalized = corrector.correct(raw_words)
                print(f"\n{'='*50}")
                print(f"  FINALIZED: {finalized}")
                print(f"{'='*50}\n")
                speak(finalized)
                sentence_b.reset()
                memory.reset()
                letter_buf.reset()

        # ── Build display state ────────────────────────────────────────
        building = sentence_b.display
        disp_sentence = finalized if not building else building

        # Grammar-correct the building sentence for live display
        if building:
            disp_sentence = corrector.correct(sentence_b.words)

        state = {
            "sentence":     disp_sentence,
            "current_pred": cur_pred,
            "confidence":   cur_conf,
            "status":       cur_status if hands_detected else "NO HANDS",
            "mode":         mode,
            "letter_buffer": letter_buf.display_string,
            "fps":          fps,
            "ar_mode":      ar_display.ar_mode,
            "suggestions":  memory.context_suggestions,
        }

        # ── Render ─────────────────────────────────────────────────────
        output = ar_display.render(frame, state)

        cv2.imshow("Sign Language Communication System", output)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if hands_proc:
        hands_proc.close()
    print("[INFO] System closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-Time Sign Language Communication System")
    parser.add_argument("--model", type=str, default="model/model_mlp.pkl")
    args = parser.parse_args()
    main(model_path=args.model)
