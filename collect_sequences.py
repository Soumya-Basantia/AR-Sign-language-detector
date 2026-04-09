"""
collect_sequences.py
--------------------
Captures webcam video and extracts MediaPipe landmark features
for a defined set of sign language words.

Each 'sequence' = SEQUENCE_LENGTH consecutive frames of landmark data.
Run this script once per word label to build your dataset.

Usage:
    python collect_sequences.py
"""

import cv2
import numpy as np
import os
import time
import json
from feature_extraction import extract_features

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# WORDS = ["I", "You", "Need", "Help", "Thank You", "Yes", "No", "Please", "Sorry", "Want","Stop","Go","Come","Call","Water","Doctor"]  # list of words to collect
SEQUENCE_LENGTH = 40          # frames per sample
SEQUENCES_PER_WORD = 30       # how many samples to collect per word
DATA_DIR = "data/sequences"   # where .npy files are saved
COLLECTION_DELAY = 2          # seconds pause before each new sequence
VOCAB_FILE = "vocabulary.json"  # vocabulary file

def load_words():
    """Load words from vocabulary file."""
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'r') as f:
            vocab = json.load(f)
            return vocab["words"]
    else:
        # Fallback to default words if file doesn't exist
        return ["I", "You", "Need", "Help", "Thank You", "Yes", "No", "Please", "Sorry", "Want","Stop","Go","Come","Call","Water","Doctor"]

WORDS = load_words()  # Load words dynamically


def ensure_dirs():
    """Create data directories for each word if they don't exist."""
    for word in WORDS:
        path = os.path.join(DATA_DIR, word)
        os.makedirs(path, exist_ok=True)
    print(f"[INFO] Data directories ready under '{DATA_DIR}/'")


def next_sequence_index(word):
    """Return the next available sequence index for a given word."""
    folder = os.path.join(DATA_DIR, word)
    existing = [f for f in os.listdir(folder) if f.endswith(".npy")]
    return len(existing)


def collect_for_word(word, cap):
    """
    Interactively collect SEQUENCES_PER_WORD sequences for one word.
    Shows a live preview with on-screen instructions.
    """
    num_done = next_sequence_index(word)
    print(f"\n[COLLECT] Word: {word}  (already have {num_done} sequences)")

    seq_idx = num_done
    while seq_idx < num_done + SEQUENCES_PER_WORD:
        sequence_frames = []

        # ── Wait / countdown before recording ──
        wait_start = time.time()
        while time.time() - wait_start < COLLECTION_DELAY:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Draw hand landmarks for visualization
            hand_result = hands_detector.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
            
            # ── Face visualization disabled for hand-only focus ──
            # face_result = face_detector.process(rgb)
            # if face_result.multi_face_landmarks:
            #     mp_draw.draw_landmarks(
            #         frame, face_result.multi_face_landmarks[0], mp_face.FACEMESH_CONTOURS,
            #         mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
            #         mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1)
            #     )
            
            remaining = COLLECTION_DELAY - (time.time() - wait_start)
            cv2.putText(frame,
                        f"WORD: {word}  |  Seq {seq_idx+1}/{num_done+SEQUENCES_PER_WORD}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"GET READY... {remaining:.1f}s",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Collect Sign Data", frame)
            cv2.waitKey(1)

        # ── Record frames ──
        frame_idx = 0
        while frame_idx < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats = extract_features(rgb, include_face=False)
            
            # Validate frame data
            if np.any(np.isnan(feats)) or np.all(feats == 0):
                print(f"[WARN] Invalid frame detected (NaN or all-zero), skipping frame {frame_idx}")
                continue
            
            sequence_frames.append(feats)

            # Draw hand landmarks for visualization
            hand_result = hands_detector.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
            
            # ── Face visualization disabled for hand-only focus ──
            # face_result = face_detector.process(rgb)
            # if face_result.multi_face_landmarks:
            #     mp_draw.draw_landmarks(
            #         frame, face_result.multi_face_landmarks[0], mp_face.FACEMESH_CONTOURS,
            #         mp_draw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
            #         mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1)
            #     )

            # Progress bar
            progress = int((frame_idx / SEQUENCE_LENGTH) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-10),
                          (progress, frame.shape[0]), (0, 255, 0), -1)
            cv2.putText(frame, f"RECORDING: {word}  [{frame_idx+1}/{SEQUENCE_LENGTH}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Collect Sign Data", frame)
            cv2.waitKey(1)
            frame_idx += 1

        # ── Save sequence ──
        if len(sequence_frames) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence_frames)  # (40, 124)
            save_path = os.path.join(DATA_DIR, word, f"{seq_idx}.npy")
            np.save(save_path, seq_array)
            print(f"  Saved sequence {seq_idx} → {save_path}")
            seq_idx += 1
        else:
            print("  [WARN] Incomplete sequence skipped.")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Collection interrupted by user.")
            return


def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check device index.")

    print("\n=== Sign Language Data Collection ===")
    print(f"Words: {WORDS}")
    print(f"Frames/seq: {SEQUENCE_LENGTH}  |  Seqs/word: {SEQUENCES_PER_WORD}")
    print("Press 'q' anytime to quit.\n")

    for word in WORDS:
        print(f"\n--- Prepare to sign: {word} ---")
        print("  Press ENTER to start collecting, or 's' to skip this word.")
        key_input = input("  > ").strip().lower()
        if key_input == "s":
            print(f"  Skipping {word}.")
            continue
        collect_for_word(word, cap)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Data collection complete!")
    print(f"       Sequences saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()
