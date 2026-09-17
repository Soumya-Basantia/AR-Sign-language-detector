# Real-Time Sign Language Communication System
### AR Glasses HUD · Gesture → Sentence · Voice Output · AI Speech Bridge

---

## Overview

A high-performance assistive communication platform that translates American Sign Language (ASL) gestures and fingerspelling into natural, fluent English speech. Features an interactive Augmented Reality (AR) glasses HUD simulation with contextual suggestions, phrase autocomplete, and real-time voice synthesis.

---

## Key Features

- **Dual Interaction Modes**:
  - **WORD Mode**: Whole-word gesture recognition using sliding-window temporal classification.
  - **LETTER Mode**: Real-time fingerspelling accumulation with Levenshtein-distance fuzzy spell correction and pause auto-flush.
- **Augmented Reality HUD**:
  - **Normal UI**: Structured control dashboard with confidence meters, live buffers, and context suggestions.
  - **AR Glasses Mode**: Minimal floating HUD, holographic subtitle animation, subtle vignette, and dynamic scanlines.
- **Optimized Computer Vision Pipeline**:
  - Direct single-pass MediaPipe processing (eliminating redundant inferences for high FPS).
  - Wrist-relative landmark normalization ($x, y$ coordinates invariant to hand screen position and distance).
  - ROI-blended UI overlays and precomputed radial gradient masks.
- **Intelligent Grammar & NLP**:
  - Automatic sentence casing, noise word filtering, and duplicate removal.
  - Topic-comment sign reordering and natural phrase expansion (e.g. `["NEED", "WATER"]` → `"I need water."`).
  - Context-aware bigram next-word prediction and phrase templates.
- **Voice Output**:
  - Thread-safe, non-blocking TTS engine queue for smooth Windows speech output on finalization.

---

## Quick Start

```bash
# 1. Install dependencies
pip install mediapipe opencv-python numpy scikit-learn pyttsx3 joblib

# 2. Run system (launches in DEMO mode if no weights present)
python predict_sequence.py

# Or load a specific model
python predict_sequence.py --model model/model_mlp.pkl
```

---

## Controls & Key Bindings

| Key | Action |
|---|---|
| `SPACE` | Accept current prediction into sentence / flush letter buffer |
| `S` | Insert space between words (LETTER mode) |
| `D` | Delete last word or letter |
| `U` | Undo last action |
| `1`, `2`, `3` | Select contextual word suggestion |
| `M` | Toggle **WORD** ↔ **LETTER** mode |
| `ENTER` | Finalize sentence → grammar correct → synthesize speech |
| `G` | Toggle **AR Glasses HUD** ↔ **Normal Dashboard** |
| `R` | Full system reset |
| `Q` | Quit application |

---

## Architecture

```
Webcam Feed (OpenCV)
  └─► MediaPipe Hands (21 keypoints × 2 hands = 84 normalized features)
        └─► Sliding Window Buffer (40 frames)
              └─► Classification Engine (Scikit-Learn MLP / LSTM)
                    └─► PredictionMemory
                    │     ├── Majority voting over temporal window (7 frames)
                    │     ├── Stability gate (5 stable frames before ready)
                    │     ├── Context boost (bigram transition table)
                    │     └── Confidence threshold gating (0.85)
                    │
                    ├─► [WORD Mode] SentenceBuilder
                    │     ├── Rolling word buffer with duplicate cooldown
                    │     └── Dynamic phrase expansion
                    │
                    └─► [LETTER Mode] LetterBuffer
                          ├── Duplicate suppression & pause-flush (1.5s)
                          └── Levenshtein fuzzy spell correction
                                └─► GrammarCorrector
                                      ├── Rule-based reordering (ASL → English)
                                      ├── Subject-verb agreement & contractions
                                      └── Sentence-case & terminal punctuation
                                            └─► ARDisplay
                                                  ├── Normal Dashboard or AR Floating HUD
                                                  └─► Non-blocking Speech Synthesis (pyttsx3)
```

---

## Dataset Collection & Model Training

### 1. Collect Custom Sequences
```bash
# Collect gesture sequences for words
python collect_sequences.py

# Collect fingerspelling letters
python collect_letters.py
```

### 2. Train Models
```bash
# Train word model (MLP default or LSTM)
python train_model.py
python train_model.py --model lstm

# Train letter model
python train_letter.py
```

### 3. Dynamic Vocabulary Management
```bash
python dynamic_trainer.py
```

---

## Automated Verification

Run the built-in unit test suite to verify grammar correction, spelling buffers, and prediction memory:

```bash
python test_system.py
```

---

## Confidence Indicators

| Color | Threshold | Meaning |
|---|---|---|
| 🟢 Green | $\ge 80\%$ | High confidence — safe to accept |
| 🟠 Orange | $60\% - 79\%$ | Medium confidence — stabilizing |
| 🔴 Red | $< 60\%$ | Low confidence / No hands detected |
