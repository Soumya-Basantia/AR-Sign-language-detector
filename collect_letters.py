"""
collect_letters.py
--------------------
Captures webcam video and extracts MediaPipe landmark features
for a defined set of sign language letters.

Each 'letter' = SEQUENCE_LENGTH consecutive frames of landmark data.
Run this script once per letter label to build your dataset.

Usage:
    python collect_letters.py
"""

import cv2
import numpy as np
import os
import time
import argparse
import mediapipe as mp
from feature_extraction import extract_features, extract_landmarks_from_results

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WORDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 26 letters
SEQUENCE_LENGTH = 40          # frames per sample
SEQUENCES_PER_WORD = 30       # how many samples to collect per word
DATA_DIR = "data_letters/sequences"   # where .npy files are saved
COLLECTION_DELAY = 2          # seconds pause before each new sequence

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)


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


def collect_for_word(word, cap, include_face=False):
    """
    Interactively collect SEQUENCES_PER_WORD sequences for one letter.
    Shows a live preview with on-screen instructions.
    """
    num_done = next_sequence_index(word)
    print(f"\n[COLLECT] Letter: {word}  (already have {num_done} sequences)")

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

            hand_result = hands_detector.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            remaining = COLLECTION_DELAY - (time.time() - wait_start)
            cv2.putText(frame,
                        f"LETTER: {word}  |  Seq {seq_idx+1}/{num_done+SEQUENCES_PER_WORD}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"GET READY... {remaining:.1f}s",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Collect Letter Data", frame)
            cv2.waitKey(1)

        # ── Record frames ──
        frame_idx = 0
        while frame_idx < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_result = hands_detector.process(rgb)
            feats, detected = extract_landmarks_from_results(
                hand_result, include_face=include_face
            )

            # Validate frame data
            if not detected or np.any(np.isnan(feats)) or np.all(feats == 0):
                print(f"[WARN] No hands or invalid frame detected, skipping frame {frame_idx}")
                continue

            sequence_frames.append(feats)

            # Draw hand landmarks for visualization
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            # Progress bar
            progress = int((frame_idx / SEQUENCE_LENGTH) * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0]-10),
                          (progress, frame.shape[0]), (0, 255, 0), -1)
            cv2.putText(frame, f"RECORDING: {word}  [{frame_idx+1}/{SEQUENCE_LENGTH}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Collect Letter Data", frame)
            cv2.waitKey(1)
            frame_idx += 1

        # ── Save sequence ──
        if len(sequence_frames) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence_frames)
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
    parser = argparse.ArgumentParser(description="Collect sign language letters dataset")
    parser.add_argument("--include-face", action="store_true", help="Include face landmarks (124 dims)")
    args = parser.parse_args()

    ensure_dirs()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check device index.")

    print("\n=== Sign Language Letter Data Collection ===")
    print(f"Letters: {WORDS}")
    print(f"Frames/seq: {SEQUENCE_LENGTH}  |  Seqs/letter: {SEQUENCES_PER_WORD}")
    print(f"Include face: {args.include_face}")
    print("Press 'q' anytime to quit.\n")

    for word in WORDS:
        print(f"\n--- Prepare to sign letter: {word} ---")
        print("  Press ENTER to start collecting, or 's' to skip this letter.")
        key_input = input("  > ").strip().lower()
        if key_input == "s":
            print(f"  Skipping {word}.")
            continue
        collect_for_word(word, cap, include_face=args.include_face)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Letter data collection complete!")
    print(f"       Sequences saved to: {DATA_DIR}/")


if __name__ == "__main__":
    main()
