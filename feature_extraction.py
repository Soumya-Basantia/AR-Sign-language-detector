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

def extract_landmarks_from_results(hand_result, face_result=None, include_face=True, normalize=False):
    """
    Extract feature vector directly from existing MediaPipe result objects.
    Eliminates redundant processing when MediaPipe was already run on the frame.

    Args:
        hand_result: MediaPipe Hands process result
        face_result: Optional MediaPipe FaceMesh process result
        include_face: If True, include face landmarks (124 dims), else hands only (84 dims)
        normalize: If True, center hand coordinates on wrist and scale by palm size

    Returns:
        (features_array, hands_detected_bool)
    """
    features = []
    left_hand = np.zeros(42, dtype=np.float32)   # 21 points × (x, y)
    right_hand = np.zeros(42, dtype=np.float32)
    hands_detected = False

    if hand_result and hand_result.multi_hand_landmarks:
        hands_detected = True
        for hand_lm, hand_info in zip(
            hand_result.multi_hand_landmarks,
            hand_result.multi_handedness
        ):
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            lms = hand_lm.landmark

            if normalize:
                # Wrist is index 0
                x0, y0 = lms[0].x, lms[0].y
                # Middle finger MCP is index 9 (stable reference for palm size)
                x9, y9 = lms[9].x, lms[9].y
                scale = float(np.hypot(x9 - x0, y9 - y0))
                if scale < 1e-4:
                    scale = 1.0

                coords = []
                for lm in lms:
                    coords.extend([(lm.x - x0) / scale, (lm.y - y0) / scale])
            else:
                coords = []
                for lm in lms:
                    coords.extend([lm.x, lm.y])

            if label == "Left":
                left_hand = np.array(coords, dtype=np.float32)
            else:
                right_hand = np.array(coords, dtype=np.float32)

    features.extend(left_hand)
    features.extend(right_hand)

    # ── Face (optional) ───────────────────
    if include_face:
        face_pts = np.zeros(len(IMPORTANT_FACE_POINTS) * 2, dtype=np.float32)
        if face_result and face_result.multi_face_landmarks:
            lms = face_result.multi_face_landmarks[0].landmark
            coords = []
            for idx in IMPORTANT_FACE_POINTS:
                coords.extend([lms[idx].x, lms[idx].y])
            face_pts = np.array(coords, dtype=np.float32)
        features.extend(face_pts)

    return np.array(features, dtype=np.float32), hands_detected


def extract_features(frame_rgb, include_face=True, normalize=False):
    """
    Run MediaPipe on one RGB frame and return a flat numpy feature vector.

    Args:
        frame_rgb: RGB frame from cv2
        include_face: If True, include face landmarks (124 dims), else hands only (84 dims)
        normalize: If True, apply wrist-relative coordinate normalization

    Returns:
        numpy array of shape (84,) or (124,)
    """
    hand_result = hands_detector.process(frame_rgb)
    face_result = face_detector.process(frame_rgb) if include_face else None
    feats, _ = extract_landmarks_from_results(
        hand_result, face_result, include_face=include_face, normalize=normalize
    )
    return feats


def extract_features_with_detection(frame_rgb, include_face=True, normalize=False):
    """
    Run MediaPipe and return both the feature vector and whether hands were detected.
    """
    hand_result = hands_detector.process(frame_rgb)
    face_result = face_detector.process(frame_rgb) if include_face else None
    return extract_landmarks_from_results(
        hand_result, face_result, include_face=include_face, normalize=normalize
    )