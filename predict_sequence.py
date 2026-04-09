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
  SPACE  -> Accept current prediction into sentence / flush letter buffer
  D      -> Delete last word / last letter
  R      -> Full reset
  M      -> Toggle WORD <-> LETTER mode
  ENTER  -> Finalize sentence (grammar correct + speak)
  G      -> Toggle AR glasses mode <-> Normal UI
  Q      -> Quit

Run
---
  python predict_sequence.py --model models/your_model.pkl

If no model is available, the system launches in DEMO mode with
synthetic predictions so all UI features can be evaluated.
"""

import argparse
import time
import joblib
import sys
import os
import threading
from collections import deque
from typing import Optional, List, Deque
from datetime import datetime

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
    import winsound
except ImportError:
    pass

try:
    import pickle
    _PICKLE_AVAILABLE = True
except ImportError:
    _PICKLE_AVAILABLE = False

# Local modules
from feature_extraction import extract_features
from letter_buffer import LetterBuffer
from grammar_corrector import GrammarCorrector
from prediction_memory import PredictionMemory
from ar_display import ARDisplay

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE        = 40       # frames per prediction window
FRAME_SKIP         = 2        # process every Nth frame
CONF_THRESHOLD     = 0.85
SENTENCE_MAX_WORDS = 10
FPS_SMOOTH_WINDOW  = 30

GLOBAL_COOLDOWN    = 1.5      # unified cooldown in seconds

USER_ACCEPTANCE_CONFIDENCE = 0.85  # threshold for accepting predictions

DEMO_WORDS  = ["HELP", "WATER", "PLEASE", "MORE", "THANK", "YOU",
               "NEED", "FOOD", "STOP", "DOCTOR"]
DEMO_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

DEMO_SENTENCES = [
    "I am hungry",
    "Need help",
    "Please help me",
    "I need water",
    "Thank you"
]


def get_status_message(status: str, conf: float, threshold: float) -> str:
    """Provide user-friendly status messages."""
    if status == "READY":
        return "[READY] Ready to accept"
    elif status == "STABILIZING":
        return "[STABILIZING] Stabilizing prediction..."
    elif status == "LOW_CONF":
        return f"[LOW_CONF] Confidence {conf:.0%} (need {threshold:.0%})"
    elif status == "COOLDOWN":
        return "[COOLDOWN] Cooldown active"
    elif status == "NO HANDS":
        return "[NO_HANDS] No hands detected"
    else:
        return f"[{status}]"


class UndoStack:
    """Simple undo stack for sentences."""

    def __init__(self, max_history: int = 10):
        self._history: List[List[str]] = []
        self._max = max_history
    
    def push(self, words: List[str]):
        self._history.append(words.copy())
        if len(self._history) > self._max:
            self._history.pop(0)
    
    def undo(self) -> Optional[List[str]]:
        if len(self._history) > 1:
            self._history.pop()
            return self._history[-1].copy()
        return None

    def clear(self):
        self._history.clear()
    
    def can_undo(self) -> bool:
        return len(self._history) > 1
class SentenceBuilder:
    """Rolling word buffer with duplicate guard."""

    def __init__(self, max_words: int = SENTENCE_MAX_WORDS,
                 cooldown: float = GLOBAL_COOLDOWN):
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

    def set_words(self, words: List[str]):
        self._words = [w.strip().upper() for w in words if w.strip()]
        self._last_word = self._words[-1] if self._words else ""

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

    def __init__(self, model= "word"):
        self._model = None
        self._classes = None
        self._valid_model = False
        self._demo_mode = False
        self.load_model(model)
    
    def load_model(self, model="word"):
        # Clean up old model first
        if self._model is not None:
            del self._model
            del self._classes
        self._valid_model = False
        
        try:
            if model == "word":
                model_path = "model/model_mlp.pkl"
                encoder_path = "model/label_encoder.pkl"
                demo_classes = DEMO_WORDS
            elif model == "letter":
                model_path = "model/model_letter.pkl"
                encoder_path = "model/label_encoder_letter.pkl"
                demo_classes = DEMO_LETTERS
            else:
                raise ValueError(f"Unknown model type: {model}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self._model = joblib.load(model_path)
            self._classes = joblib.load(encoder_path).classes_
            self._valid_model = True
            self._demo_mode = False
            print(f"[INFO] Loaded {model.upper()} model with {len(self._classes)} classes")
            
        except Exception as e:
            print(f"[ERROR] Failed to load {model} model: {e}")
            print(f"[INFO] Falling back to DEMO mode")
            self._model = None
            self._classes = np.array(demo_classes)
            self._valid_model = False
            self._demo_mode = True
    
    def __del__(self):
        """Cleanup on garbage collection."""
        if self._model is not None:
            del self._model
        
    def predict(self, window: np.ndarray):
    
        try:
            x = np.array(window.flatten(), dtype=np.float32).reshape(1, -1)
            x = np.nan_to_num(x)

            if not self._valid_model or self._model is None:
                return "---", 0.0

            if hasattr(self._model, "n_features_in_"):
                expected = int(self._model.n_features_in_)
                if x.shape[1] != expected:
                    print(f"[ERROR] Prediction failed: model expects {expected} features, but input has {x.shape[1]}.")
                    print("[INFO] Please retrain the model with the current hand-only feature extractor.")
                    self._valid_model = False
                    self._model = None
                    return "---", 0.0

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
            self._model = None
            return "---", 0.0

    @property
    def demo_mode(self):
        return self._demo_mode

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

    # ── Face excluded for hand-only prediction ──
    # face_result = face_detector.process(frame_rgb)
    # face_pts = np.zeros(len(IMPORTANT_FACE_POINTS) * 2)
    # if face_result.multi_face_landmarks:
    #     lms = face_result.multi_face_landmarks[0].landmark
    #     coords = []
    #     for idx in IMPORTANT_FACE_POINTS:
    #         coords.extend([lms[idx].x, lms[idx].y])
    #     face_pts = np.array(coords)

    # Combine both hands → 84 features (hands only)
    return np.concatenate([left_hand, right_hand])

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

def play_beep(frequency: int = 800, duration: int = 200):
    """Audio feedback disabled."""
    return
# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(model_path: Optional[str] = None):
    
    last_added = ""
    last_time = 0
    
    _init_tts()

    # Components
    predictor   = Predictor(model="word")
    memory      = PredictionMemory(window=10, min_votes=6,
                                   stable_frames=7,
                                   confidence_threshold=0.5)
    letter_buf  = LetterBuffer(pause_seconds=1.5)
    sentence_b  = SentenceBuilder(max_words=SENTENCE_MAX_WORDS)
    corrector   = GrammarCorrector(add_punctuation=True)
    ar_display  = ARDisplay()
    undo_stack  = UndoStack()

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

    # Status message for GUI feedback
    status_message: str = ""
    last_status_time: float = 0.0

    print("\n" + "="*60)
    print("  SIGN LANGUAGE COMMUNICATION SYSTEM — READY")
    print("="*60)
    print("  Controls:")
    print("    SPACE  -> Accept word/letter")
    print("    S      -> Add space (LETTER mode)")
    print("    D      -> Delete last")
    print("    U      -> Undo last action")
    print("    R      -> Reset")
    print("    M      -> Toggle mode (WORD/LETTER)")
    print("    ENTER  -> Finalize + speak")
    print("    G      -> Toggle AR glasses mode")
    print("    Q      -> Quit")
    print("="*60 + "\n")
    print("NOTE: Click on the OpenCV window to focus it for keyboard input.\n")

    window_name = "Sign Language Communication System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Mouse callback removed - using keyboard for suggestions

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
                lm = extract_features(rgb, include_face=(mode == "LETTER"))
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
                window_arr = np.array(landmark_window)   # (40, 84) - hands only
                raw_label, raw_conf = predictor.predict(window_arr)
                memory.push(raw_label, raw_conf)

                stable_label, eff_conf, status = memory.get_stable_prediction()
                cur_pred   = stable_label or raw_label
                cur_conf   = round(eff_conf, 2)
                cur_status = status

                if not predictor._valid_model:
                    cur_pred = "---"
                    cur_conf = 0.0
                    cur_status = "NO MODEL"
                
            # auto-accept logic for WORD mode
                
                if mode == "WORD" and cur_status == "READY" and cur_conf >= 0.85:
                    
                    if time.time() - last_time > GLOBAL_COOLDOWN:# 1 second cooldown
                        last_time = time.time()
                    
                        if cur_pred != last_added:
                            accepted = sentence_b.add(cur_pred)
                            if accepted:
                                memory.confirm(cur_pred)
                                print(f"[WORD] Added '{cur_pred}' → "
                                    f"{sentence_b.display}")
                                last_added = cur_pred
                
            # auto-add logic for LETTER mode
            elif mode == "LETTER" and cur_status == "READY" and cur_conf >= USER_ACCEPTANCE_CONFIDENCE:
                if time.time() - last_time > GLOBAL_COOLDOWN:
                    last_time = time.time()
                    accepted = letter_buf.add_letter(cur_pred)
                    if accepted:
                        print(f"[LETTER-AUTO] Added '{cur_pred}' → {letter_buf.display_string}")
                    else:
                        print(f"[LETTER-REJECTED] '{cur_pred}' (duplicate or buffer full)")
                        last_added = cur_pred
                
            elif not hands_detected:
                cur_status = "NO HANDS"
                cur_pred   = ""
                cur_conf   = 0.0
                last_added = ""
            else:
                cur_status = "STABILIZING"

        # ── Key handling ───────────────────────────────────────────────
        key = cv2.waitKeyEx(1)

        # Debug: print key presses
        if key != -1:
            print(f"Key pressed: {key} ({chr(key) if 32 <= key <= 126 else 'special'})")

        # Clear status message after 2 seconds
        if time.time() - last_status_time > 2:
            status_message = ""

        if key == ord('q'):
            break

        elif key in (ord('g'), ord('G')):
            ar_display.toggle_mode()
            m = "AR GLASSES" if ar_display.ar_mode else "NORMAL"
            print(f"[UI] Switched to {m} mode")
            status_message = f"AR MODE: {'ON' if ar_display.ar_mode else 'OFF'}"
            last_status_time = time.time()

        elif key in (ord('r'), ord('R')):
            play_beep(500, 200)  # Reset beep
            sentence_b.reset()
            memory.reset()
            letter_buf.reset()
            undo_stack.clear()
            finalized = ""  # Clear finalized display
            print("[RESET] Sentence cleared")
            status_message = "SENTENCE RESET"
            last_status_time = time.time()

        elif key == ord('1'):
            suggestions = memory.context_suggestions
            if suggestions and len(suggestions) >= 1:
                suggestion = suggestions[0]
                if mode == "LETTER" and not letter_buf.is_empty:
                    flushed = letter_buf.flush_on_space()
                    for word in flushed:
                        sentence_b.add(word)
                    letter_buf.reset()
                if sentence_b.add(suggestion):
                    memory.confirm(suggestion)
                    undo_stack.push(sentence_b.words)
                    print(f"[SUGGESTION] Key 1: added '{suggestion}'")
                else:
                    print(f"[SUGGESTION] Key 1: '{suggestion}' could not be added")

        elif key == ord('2'):
            suggestions = memory.context_suggestions
            if suggestions and len(suggestions) >= 2:
                suggestion = suggestions[1]
                if mode == "LETTER" and not letter_buf.is_empty:
                    flushed = letter_buf.flush_on_space()
                    for word in flushed:
                        sentence_b.add(word)
                    letter_buf.reset()
                if sentence_b.add(suggestion):
                    memory.confirm(suggestion)
                    undo_stack.push(sentence_b.words)
                    print(f"[SUGGESTION] Key 2: added '{suggestion}'")
                else:
                    print(f"[SUGGESTION] Key 2: '{suggestion}' could not be added")

        elif key == ord('3'):
            suggestions = memory.context_suggestions
            if suggestions and len(suggestions) >= 3:
                suggestion = suggestions[2]
                if mode == "LETTER" and not letter_buf.is_empty:
                    flushed = letter_buf.flush_on_space()
                    for word in flushed:
                        sentence_b.add(word)
                    letter_buf.reset()
                if sentence_b.add(suggestion):
                    memory.confirm(suggestion)
                    undo_stack.push(sentence_b.words)
                    print(f"[SUGGESTION] Key 3: added '{suggestion}'")
                else:
                    print(f"[SUGGESTION] Key 3: '{suggestion}' could not be added")

        elif key in (ord('m'), ord('M')):
            if mode == "WORD":
                mode = "LETTER"
                predictor.load_model("letter")
                letter_buf.reset()
                sequence_frames.clear()  # Clear buffer for dimension change
                print("[MODE] Switched to LETTER mode")
            else:
                mode = "WORD"
                predictor.load_model("word")
                letter_buf.reset()
                sequence_frames.clear()  # Clear buffer for dimension change
                print("[MODE] Switched to WORD mode")

        elif key == ord('s'):
            if mode == "LETTER":
                play_beep(900, 150)  # Space beep
                letter_buf.add_space()  # Add space to buffer
                print(f"[LETTER] Added space → {letter_buf.display_string}")
            else:
                print("[SKIP] Space only available in LETTER mode")

        elif key == ord('d'):
            play_beep(600, 200)  # Delete beep
            if mode == "LETTER" and not letter_buf.is_empty:
                removed = letter_buf.delete_last()
                print(f"[DELETE] Removed letter '{removed}'")
            else:
                removed = sentence_b.delete_last()
                memory.delete_last_confirmed()
                print(f"[DELETE] Removed word '{removed}'")

        elif key == ord('u'):
            if undo_stack.can_undo():
                play_beep(800, 200)  # Undo beep
                restored = undo_stack.undo()
                if restored:
                    sentence_b._words = restored
                    print(f"[UNDO] Restored sentence: {' '.join(restored)}")
            else:
                play_beep(400, 300)  # Error beep
                print("[UNDO] Nothing to undo")

        elif key == 32:  # SPACE
            if cur_status == "READY" and cur_conf >= USER_ACCEPTANCE_CONFIDENCE:
                play_beep(1000, 150)  # Success beep
                if mode == "LETTER":
                    # In letter mode, space flushes the current letter buffer as words
                    if not letter_buf.is_empty:
                        flushed_words = letter_buf.flush_on_space()
                        for word in flushed_words:
                            if word.strip():  # Skip empty words
                                sentence_b.add(word.strip())
                                memory.confirm(word.strip())
                        if flushed_words:
                            print(f"[LETTER-FLUSH] Added words: {flushed_words} → {sentence_b.display}")
                else:
                    accepted = sentence_b.add(cur_pred)
                    if accepted:
                        memory.confirm(cur_pred)
                        print(f"[WORD] Added '{cur_pred}' → {sentence_b.display}")
            else:
                play_beep(400, 300)  # Error beep
                print(f"[SKIP] status={cur_status} conf={cur_conf:.2f}")

        elif key == 13:  # ENTER
            if mode == "LETTER" and not letter_buf.is_empty:
                # Flush remaining letters as a word first
                flushed = letter_buf.flush_on_space()
                if flushed:
                    sentence_b.add(flushed)

            raw_words = sentence_b.words
            if raw_words:
                play_beep(1200, 300)  # Finalize beep
                undo_stack.push(raw_words)  # Save current state for undo
                finalized = corrector.correct(raw_words)
                print(f"\n{'='*50}")
                print(f"  FINALIZED: {finalized}")
                print(f"{'='*50}\n")
                speak(finalized)
                
                # Save finalized sentence to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data = {"timestamp": timestamp, "sentence": finalized, "raw_words": raw_words}
                os.makedirs("data", exist_ok=True)
                with open(f"data/finalized_{timestamp}.json", "w") as f:
                    json.dump(data, f)
                
                sentence_b.reset()
                memory.reset()
                letter_buf.reset()
            else:
                play_beep(400, 300)  # Error beep

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
            "status":       get_status_message(cur_status, cur_conf, CONF_THRESHOLD) if hands_detected else "NO HANDS",
            "mode":         mode,
            "letter_buffer": letter_buf.display_string,
            "fps":          fps,
            "ar_mode":      ar_display.ar_mode,
            "suggestions":  memory.context_suggestions,
            "phrase_suggestions": memory.phrase_suggestions(sentence_b.words),
            "demo_mode":    predictor.demo_mode,
            "status_message": status_message,
        }

        # ── Render ─────────────────────────────────────────────────────
        output = ar_display.render(frame, state)

        cv2.imshow(window_name, output)

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
