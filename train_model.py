"""
train_model.py
--------------
Loads the collected sequence data, trains a classification model,
and saves the trained model + label encoder to disk.

Two backends are supported:
  - 'mlp'  : Fast MLP (scikit-learn) — default, great for small datasets
  - 'lstm' : LSTM (TensorFlow/Keras) — better accuracy with more data

Usage:
    python train_model.py                    # uses MLP
    python train_model.py --model lstm       # uses LSTM
"""

import os
import argparse
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import json

DATA_DIR  = "data/sequences"
MODEL_DIR = "model"
VOCAB_FILE = "vocabulary.json"

def load_words():
    """Load words from vocabulary file."""
    if os.path.exists(VOCAB_FILE):
        with open(VOCAB_FILE, 'r') as f:
            vocab = json.load(f)
            return vocab["words"]
    else:
        # Fallback to default words if file doesn't exist
        return ["I", "You", "Need", "Help", "Thank You", "Yes", "No", "Please", "Sorry", "Want","Stop","Go","Come","Call","Water","Doctor"]

WORDS     = load_words()  # Load words dynamically

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def load_dataset(flatten=True):
    """
    Load all .npy sequences from DATA_DIR.

    Args:
        flatten: If True, each (40, 84) sequence → 3360-dim vector.
                 If False, keep shape (40, 84) for LSTM.

    Returns:
        X: numpy array of features
        y: numpy array of string labels
    """
    X, y = [], []
    missing = []

    for word in WORDS:
        folder = os.path.join(DATA_DIR, word)
        if not os.path.isdir(folder):
            missing.append(word)
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        if not files:
            missing.append(word)
            continue
        for fname in files:
            seq = np.load(os.path.join(folder, fname))  # (40, 124)
            if flatten:
                X.append(seq.flatten())
            else:
                X.append(seq)
            y.append(word)

    if missing:
        print(f"[WARN] No data found for words: {missing}")
    print(f"[DATA] Loaded {len(X)} sequences across {len(set(y))} words.")
    return np.array(X), np.array(y)


def augment_sequences(X, y, factor=2):
    """
    Simple augmentation: add small Gaussian noise to existing sequences.
    Helps with small datasets.
    """
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.005, X.shape).astype(np.float32)
        X_aug.append(X + noise)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)


# ─────────────────────────────────────────────
# MLP Training (scikit-learn)
# ─────────────────────────────────────────────
def train_mlp(X_train, X_test, y_train, y_test):
    """
    Train a Multi-Layer Perceptron classifier.
    Fast, reliable, works well with 200-500 samples per class.
    """
    print("\n[TRAIN] Training MLP classifier...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    
    X_train = np.stack(X_train)
    X_test  = np.stack(X_test)
    
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    X_train = np.nan_to_num(X_train)
    X_test  = np.nan_to_num(X_test)

    print("X_train dtype:", X_train.dtype)
    print("NaN count:", np.isnan(X_train).sum())
    print("Inf count:", np.isinf(X_train).sum())

    model.fit(X_train, y_train)
    
    return model


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest — good baseline for very small datasets.
    """
    print("\n[TRAIN] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# LSTM Training (TensorFlow/Keras)
# ─────────────────────────────────────────────
def train_lstm(X_train, X_test, y_train_enc, y_test_enc, num_classes):
    """
    Train a lightweight LSTM for sequence classification.
    Requires TensorFlow. Each sample shape: (sequence_len, features).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                              BatchNormalization, Input)
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    print(f"\n[TRAIN] Training LSTM model (TF {tf.__version__})...")

    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat  = to_categorical(y_test_enc,  num_classes)

    seq_len, n_features = X_train.shape[1], X_train.shape[2]

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1)
    ]

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150,
        batch_size=16,
        callbacks=callbacks
    )
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, label_encoder, model_type):
    """Print classification report and per-class accuracy."""
    if model_type == "lstm":
        probs = model.predict(X_test)
        y_pred_enc = probs.argmax(axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        y_true = label_encoder.inverse_transform(y_test)
    else:
        y_pred = model.predict(X_test)
        y_true = y_test

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred))

    print("CONFUSION MATRIX")
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm,
                         index=label_encoder.classes_,
                         columns=label_encoder.classes_)
    print(df_cm.to_string())
    return y_pred


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train sign language model")
    parser.add_argument("--model", choices=["mlp", "rf", "lstm"],
                        default="mlp", help="Model backend to use")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation (recommended for small datasets)")
    args = parser.parse_args()

    use_lstm = (args.model == "lstm")

    # ── Load data ──
    X, y = load_dataset(flatten=not use_lstm)
    
    import numpy as np
    
    X = np.array(X, dtype=np.float32)
    
    X = np.nan_to_num(X)
    X = np.clip(X, -1, 1)

    if len(X) == 0:
        print("[ERROR] No data found. Run collect_sequences.py first.")
        return

    # ── Encode labels ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_enc = np.array(y_enc)
    
    print("y dtype:", y_enc.dtype)
    print("Unique labels:", np.unique(y_enc))
    print(f"[INFO] Classes: {list(le.classes_)}")

    # ── Augment (flatten mode only) ──
    if args.augment and not use_lstm:
        print("[INFO] Applying data augmentation (×3)...")
        X, y = augment_sequences(X, y, factor=3)
        y_enc = le.transform(y)
        print(f"[INFO] Dataset size after augmentation: {len(X)}")

    # ── Train / test split ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )
    print(f"[INFO] Train: {len(X_tr)}  Test: {len(X_te)}")

    # ── Train ──
    if args.model == "mlp":
        model = train_mlp(X_tr, X_te, y_tr, y_te)
        evaluate(model, X_te, y_te, le, "mlp")
        joblib.dump(model, os.path.join(MODEL_DIR, "model_mlp.pkl"))
        print(f"\n[SAVED] model/model_mlp.pkl")

    elif args.model == "rf":
        model = train_random_forest(X_tr, X_te, y_tr, y_te)
        evaluate(model, X_te, y_te, le, "rf")
        joblib.dump(model, os.path.join(MODEL_DIR, "model_rf.pkl"))
        print(f"\n[SAVED] model/model_rf.pkl")

    elif args.model == "lstm":
        model = train_lstm(X_tr, X_te, y_tr, y_te, len(le.classes_))
        evaluate(model, X_te, y_te, le, "lstm")
        model.save(os.path.join(MODEL_DIR, "model_lstm.keras"))
        print(f"\n[SAVED] model/model_lstm.keras")

    # ── Save label encoder + metadata ──
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    meta = {
        "words": list(le.classes_),
        "model_type": args.model,
        "sequence_length": X.shape[1] if not use_lstm else X.shape[1],
        "feature_size": X.shape[-1]
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVED] model/label_encoder.pkl")
    print(f"[SAVED] model/metadata.json")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
