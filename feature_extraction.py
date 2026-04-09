"""
feature_extraction.py
----------------------
Shared module for extracting MediaPipe landmark features for sign language detection.

Supports both hand-only (84 dims) and hand+face (124 dims) feature vectors.
"""

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────
# MediaPipe setup (shared across calls)
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

face_detector = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    refine_landmarks=False
)

# Face landmark indices we care about (fewer = faster)
IMPORTANT_FACE_POINTS = [0, 1, 4, 9, 13, 14, 17, 61, 78, 291, 308,
                         33, 133, 362, 263, 168, 6, 197, 195, 5]

def extract_features(frame_rgb, include_face=True):
    """
    Run MediaPipe on one RGB frame and return a flat numpy feature vector.

    Args:
        frame_rgb: RGB frame from cv2
        include_face: If True, include face landmarks (124 dims), else hands only (84 dims)

    Returns:
        numpy array of shape (84,) or (124,) with normalized landmark coordinates
    """
    features = []

    # ── Hands ──────────────────────────────
    hand_result = hands_detector.process(frame_rgb)
    left_hand = np.zeros(42)   # 21 points × (x, y)
    right_hand = np.zeros(42)

    if hand_result.multi_hand_landmarks:
        for hand_lm, hand_info in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness
        ):
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            coords = []
            for lm in hand_lm.landmark:
                coords.extend([lm.x, lm.y])
            if label == "Left":
                left_hand = np.array(coords)
            else:
                right_hand = np.array(coords)

    features.extend(left_hand)
    features.extend(right_hand)

    # ── Face (optional) ───────────────────
    if include_face:
        face_result = face_detector.process(frame_rgb)
        face_pts = np.zeros(len(IMPORTANT_FACE_POINTS) * 2)

        if face_result.multi_face_landmarks:
            lms = face_result.multi_face_landmarks[0].landmark
            coords = []
            for idx in IMPORTANT_FACE_POINTS:
                coords.extend([lms[idx].x, lms[idx].y])
            face_pts = np.array(coords)

        features.extend(face_pts)

    return np.array(features, dtype=np.float32)