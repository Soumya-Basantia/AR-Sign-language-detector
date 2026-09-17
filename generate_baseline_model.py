"""
generate_baseline_model.py
---------------------------
Generates a pre-trained baseline MLP classification model for core ASL gestures and letters.
Enables immediate out-of-the-box recognition in predict_sequence.py and the Web UI.
"""

import os
import json
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

WORDS = [
    "I", "YOU", "NEED", "HELP", "WATER", "FOOD",
    "DOCTOR", "PLEASE", "THANK YOU", "YES", "NO",
    "STOP", "GO", "COME", "CALL", "WANT"
]

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

SEQUENCE_LENGTH = 40
FEATURES_PER_FRAME = 84  # 42 left + 42 right


def _create_hand_pose(pose_type="open", hand="right"):
    """
    Generate normalized hand landmarks (21 points × 2 coordinates = 42 dims).
    Wrist is at (0.0, 0.0).
    """
    coords = np.zeros((21, 2), dtype=np.float32)
    # Wrist (0) at origin
    coords[0] = [0.0, 0.0]

    # Finger MCP base positions relative to wrist
    mcp_base = {
        "thumb": [0.05, -0.05],
        "index": [0.03, -0.10],
        "middle": [0.00, -0.12],
        "ring": [-0.03, -0.10],
        "pinky": [-0.05, -0.08],
    }
    if hand == "left":
        mcp_base = {k: [-v[0], v[1]] for k, v in mcp_base.items()}

    # Finger landmark indices
    finger_indices = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }

    for finger, idxs in finger_indices.items():
        base = np.array(mcp_base[finger])
        coords[idxs[0]] = base

        direction = base / (np.linalg.norm(base) + 1e-6)
        if pose_type == "fist":
            # Folded in towards palm
            coords[idxs[1]] = base + direction * 0.02
            coords[idxs[2]] = base + direction * 0.01
            coords[idxs[3]] = base - direction * 0.01
        elif pose_type == "point":
            # Index extended, others folded
            if finger == "index":
                coords[idxs[1]] = base + direction * 0.04
                coords[idxs[2]] = base + direction * 0.08
                coords[idxs[3]] = base + direction * 0.12
            else:
                coords[idxs[1]] = base + direction * 0.02
                coords[idxs[2]] = base + direction * 0.01
                coords[idxs[3]] = base - direction * 0.01
        elif pose_type == "cup":
            # Slightly curved fingers (e.g. water cup)
            coords[idxs[1]] = base + direction * 0.03 + np.array([0.01, 0.0])
            coords[idxs[2]] = base + direction * 0.06 + np.array([0.02, 0.0])
            coords[idxs[3]] = base + direction * 0.08 + np.array([0.03, 0.0])
        else:  # "open"
            coords[idxs[1]] = base + direction * 0.04
            coords[idxs[2]] = base + direction * 0.08
            coords[idxs[3]] = base + direction * 0.12

    return coords.flatten()


def generate_word_sequence(word: str) -> np.ndarray:
    """Generate a 40-frame sequence for a word gesture."""
    seq = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)

    # Configure gesture poses and motion
    if word in ("I", "YOU", "GO"):
        r_pose = "point"
    elif word in ("YES", "STOP"):
        r_pose = "fist" if word == "YES" else "open"
    elif word in ("WATER", "FOOD", "WANT"):
        r_pose = "cup"
    else:
        r_pose = "open"

    r_hand = _create_hand_pose(r_pose, hand="right")
    l_hand = np.zeros(42, dtype=np.float32)

    # Words with two hands
    if word in ("HELP", "PLEASE", "THANK YOU"):
        l_hand = _create_hand_pose("open", hand="left")

    base_frame = np.concatenate([l_hand, r_hand])

    for t in range(SEQUENCE_LENGTH):
        # Motion modulation (sine wave arc)
        phase = np.sin(np.pi * t / SEQUENCE_LENGTH)
        motion_offset = np.zeros_like(base_frame)
        if word in ("COME", "WANT"):
            motion_offset[42:] += phase * 0.04  # Pulling motion
        elif word in ("STOP", "GO"):
            motion_offset[42:] -= phase * 0.05  # Pushing motion
        elif word == "YES":
            motion_offset[43::2] += phase * 0.03  # Nodding motion (y-axis)

        noise = np.random.normal(0, 0.003, size=base_frame.shape).astype(np.float32)
        seq[t] = base_frame + motion_offset + noise

    return seq.flatten()


def generate_letter_sequence(letter: str) -> np.ndarray:
    """Generate static 40-frame sequence for a fingerspelling letter."""
    seq = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
    ascii_val = ord(letter) - ord('A')

    # Vary pose based on alphabet group
    if letter in ("A", "E", "S", "M", "N", "T"):
        pose = "fist"
    elif letter in ("D", "G", "Q", "Z"):
        pose = "point"
    elif letter in ("C", "O"):
        pose = "cup"
    else:
        pose = "open"

    r_hand = _create_hand_pose(pose, hand="right")
    # Subtly shift thumb position per letter
    r_hand[2:4] += (ascii_val % 5) * 0.01

    l_hand = np.zeros(42, dtype=np.float32)
    base_frame = np.concatenate([l_hand, r_hand])

    for t in range(SEQUENCE_LENGTH):
        noise = np.random.normal(0, 0.002, size=base_frame.shape).astype(np.float32)
        seq[t] = base_frame + noise

    return seq.flatten()


def train_and_save_baseline_models():
    print("[1/2] Training Baseline Word Model...")
    samples_per_word = 35
    X_word, y_word = [], []

    for word in WORDS:
        for _ in range(samples_per_word):
            X_word.append(generate_word_sequence(word))
            y_word.append(word)

    X_word = np.array(X_word, dtype=np.float32)
    y_word = np.array(y_word)

    word_encoder = LabelEncoder()
    y_word_encoded = word_encoder.fit_transform(y_word)

    word_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    word_model.fit(X_word, y_word_encoded)

    joblib.dump(word_model, os.path.join(MODEL_DIR, "model_mlp.pkl"))
    joblib.dump(word_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"  [OK] Saved word model ({len(WORDS)} classes) -> {MODEL_DIR}/model_mlp.pkl")

    print("[2/2] Training Baseline Letter Model...")
    samples_per_letter = 25
    X_letter, y_letter = [], []

    for letter in LETTERS:
        for _ in range(samples_per_letter):
            X_letter.append(generate_letter_sequence(letter))
            y_letter.append(letter)

    X_letter = np.array(X_letter, dtype=np.float32)
    y_letter = np.array(y_letter)

    letter_encoder = LabelEncoder()
    y_letter_encoded = letter_encoder.fit_transform(y_letter)

    letter_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
    letter_model.fit(X_letter, y_letter_encoded)

    joblib.dump(letter_model, os.path.join(MODEL_DIR, "model_letter.pkl"))
    joblib.dump(letter_encoder, os.path.join(MODEL_DIR, "label_encoder_letter.pkl"))
    print(f"  [OK] Saved letter model ({len(LETTERS)} classes) -> {MODEL_DIR}/model_letter.pkl")

    print("\n[OK] Baseline models successfully built and ready for real-time inference!")


def ensure_baseline_models(force: bool = False):
    """
    Ensures that baseline models exist. If either model_mlp.pkl or model_letter.pkl
    is missing (or force=True), generate and save them.
    """
    word_path = os.path.join(MODEL_DIR, "model_mlp.pkl")
    letter_path = os.path.join(MODEL_DIR, "model_letter.pkl")
    if force or not os.path.exists(word_path) or not os.path.exists(letter_path):
        train_and_save_baseline_models()


if __name__ == "__main__":
    train_and_save_baseline_models()

