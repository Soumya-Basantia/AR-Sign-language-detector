# Real-Time Sign Language Communication System
### AR Glasses HUD · Gesture → Sentence · Voice Output

---

## Quick Start

```bash
# Install dependencies
pip install mediapipe opencv-python numpy pyttsx3

# Run with your trained model
python predict_sequence.py --model models/your_model.pkl

# Run in DEMO mode (no model needed — all UI features work)
python predict_sequence.py
```

---

## Key Bindings

| Key     | Action                                          |
|---------|-------------------------------------------------|
| `SPACE` | Accept current prediction (word or letter)      |
| `D`     | Delete last word / last letter                  |
| `R`     | Full system reset                               |
| `M`     | Toggle WORD ↔ LETTER mode                      |
| `ENTER` | Finalize sentence → grammar correct → speak     |
| `G`     | Toggle AR Glasses ↔ Normal UI                  |
| `Q`     | Quit                                            |

---

## Architecture

```
Camera
  └─► MediaPipe Hands (21 keypoints × 3D = 63 features)
        └─► Sliding Window Buffer (40 frames)
              └─► Predictor (MLP / RF / LSTM wrapper)
                    └─► PredictionMemory
                    │     ├── Sliding window majority vote (7 frames)
                    │     ├── Stability gate (5 stable frames)
                    │     ├── Context boost (bigram table)
                    │     └── Confidence threshold (0.70)
                    │
                    ├─► [WORD mode] SentenceBuilder
                    │     ├── Rolling buffer (10 words)
                    │     ├── Duplicate cooldown guard
                    │     └── GrammarCorrector (rule-based)
                    │
                    └─► [LETTER mode] LetterBuffer
                          ├── Same-letter cooldown guard
                          ├── Auto-flush on pause (1.5 s)
                          └── AutoCorrect (Levenshtein + lookup)

Output ──► ARDisplay
             ├── Normal UI (structured dark panels)
             └── AR Glasses (floating HUD, vignette, scanlines)
                   └─► pyttsx3 voice output (on ENTER)
```

---

## Module Reference

### `predict_sequence.py`
Main entry point. Orchestrates all components.
- `Predictor` — Model wrapper with DEMO fallback
- `SentenceBuilder` — Rolling word buffer
- Frame-skip optimization (process every 2nd frame)

### `prediction_memory.py` — `PredictionMemory`
- `push(label, conf)` — Feed raw frame prediction
- `get_stable_prediction()` → `(label, conf, status)`
  - status: `"READY"` / `"STABILIZING"` / `"LOW_CONF"` / `"COOLDOWN"`
- `confirm(label)` — Lock in an accepted word
- `context_suggestions` — Next-word hints from bigram table

### `letter_buffer.py` — `LetterBuffer`
- `add_letter(letter)` — Add signed letter (duplicate guard)
- `flush_on_space()` → corrected word string
- `check_pause_flush()` — Auto-flush after silence
- `display_string` → `"H E L P"` for HUD

### `grammar_corrector.py` — `GrammarCorrector`
- `correct(words: List[str])` → polished sentence string
- Pipeline: noise strip → dedup → agreement fix → capitalise → punctuate

### `ar_display.py` — `ARDisplay`
- `render(frame, state_dict)` → annotated frame
- `toggle_mode()` — Normal ↔ AR Glasses
- AR mode: vignette, scanlines, corner brackets, floating text, pulse ring

---

## Model Integration

Your model `.pkl` file should be one of:

**Option A — Raw sklearn model**
```python
import pickle
with open("models/model.pkl", "wb") as f:
    pickle.dump(trained_model, f)
# model must have .predict_proba(X) or .predict(X)
```

**Option B — Dict with classes**
```python
with open("models/model.pkl", "wb") as f:
    pickle.dump({"model": trained_model, "classes": class_names}, f)
```

Input shape: `(1, WINDOW_SIZE × 63)` = `(1, 2520)`

---

## Confidence Color Coding

| Color  | Threshold       | Meaning                      |
|--------|-----------------|------------------------------|
| 🟢 Green  | ≥ 80%       | High confidence — safe to accept |
| 🟠 Orange | 60% – 79%   | Medium — stabilizing         |
| 🔴 Red    | < 60%       | Low — prediction unreliable  |

---

## Adding Custom Words / Corrections

Edit `letter_buffer.py`:
```python
CORRECTION_MAP = {
    "YORU": "YOUR",
    "TOMOROW": "TOMORROW",
    # ... add your own
}

COMMON_WORDS = [
    "YOUR", "TOMORROW", ...  # fuzzy match dictionary
]
```

---

## Future Extensions

- **Smart glasses**: Replace `cv2.imshow` with WebSocket stream to Android/iOS
- **Custom LSTM**: Drop any `(WINDOW_SIZE, 63)` → softmax model into `Predictor`
- **Larger dictionary**: Swap `COMMON_WORDS` for a full word list file
- **Per-user calibration**: Record normalization offsets per hand size
