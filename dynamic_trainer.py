"""
dynamic_trainer.py
------------------
Dynamic word management system for sign language recognition.
Allows adding/removing words and managing training data.
"""

import json
import os
import shutil
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from feature_extraction import extract_features

VOCAB_FILE = "vocabulary.json"
DATA_DIR = "data/sequences"
MODEL_DIR = "model"

def load_vocabulary():
    """Load vocabulary from JSON file."""
    if not os.path.exists(VOCAB_FILE):
        # Create default vocabulary if file doesn't exist
        vocab = {
            "words": ["I", "You", "Need", "Help", "Thank You", "Yes", "No", "Please", "Sorry", "Want", "Stop", "Go", "Come", "Call", "Water", "Doctor"],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0"
        }
        save_vocabulary(vocab)
        return vocab

    with open(VOCAB_FILE, 'r') as f:
        return json.load(f)

def save_vocabulary(vocab):
    """Save vocabulary to JSON file."""
    vocab["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VOCAB_FILE, 'w') as f:
        json.dump(vocab, f, indent=2)

def add_word(word):
    """Add a new word to vocabulary."""
    vocab = load_vocabulary()

    # Check if word already exists
    if word in vocab["words"]:
        print(f"❌ Word '{word}' already exists in vocabulary!")
        return False

    # Add word and create data directory
    vocab["words"].append(word)
    os.makedirs(os.path.join(DATA_DIR, word), exist_ok=True)

    save_vocabulary(vocab)
    print(f"✅ Added word '{word}' to vocabulary")
    print(f"📁 Created data directory: {DATA_DIR}/{word}")
    return True

def remove_word(word):
    """Remove a word from vocabulary and delete its data."""
    vocab = load_vocabulary()

    if word not in vocab["words"]:
        print(f"❌ Word '{word}' not found in vocabulary!")
        return False

    # Remove from vocabulary
    vocab["words"].remove(word)
    save_vocabulary(vocab)

    # Delete data directory if it exists
    word_dir = os.path.join(DATA_DIR, word)
    if os.path.exists(word_dir):
        shutil.rmtree(word_dir)
        print(f"🗑️  Deleted data directory: {word_dir}")

    print(f"✅ Removed word '{word}' from vocabulary")
    return True

def list_words():
    """Display current vocabulary."""
    vocab = load_vocabulary()
    print(f"\n📚 Current Vocabulary ({len(vocab['words'])} words):")
    print("=" * 50)
    for i, word in enumerate(vocab["words"], 1):
        word_dir = os.path.join(DATA_DIR, word)
        count = len([f for f in os.listdir(word_dir) if f.endswith('.npy')]) if os.path.exists(word_dir) else 0
        print(f"{i:2d}. {word:<15} ({count} sequences)")
    print(f"\n📅 Last updated: {vocab['last_updated']}")

def collect_word_data(word):
    """Collect data for a specific word."""
    vocab = load_vocabulary()

    if word not in vocab["words"]:
        print(f"❌ Word '{word}' not in vocabulary! Add it first.")
        return

    print(f"🎥 Starting data collection for: {word}")
    print("Make sure your webcam is ready...")

    # Import here to avoid loading OpenCV unnecessarily
    import subprocess
    import sys

    # Create a temporary script for single word collection
    temp_script = f"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Configuration
SEQUENCE_LENGTH = 40
SEQUENCES_PER_WORD = 30
COLLECTION_DELAY = 2
DATA_DIR = "{DATA_DIR}"

def collect_for_word(word):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    # Count existing sequences
    word_dir = os.path.join(DATA_DIR, word)
    existing = [f for f in os.listdir(word_dir) if f.endswith(".npy")]
    seq_idx = len(existing)

    print(f"Already have {{seq_idx}} sequences for '{{word}}'")

    while seq_idx < SEQUENCES_PER_WORD:
        sequence_frames = []

        # Countdown
        wait_start = time.time()
        while time.time() - wait_start < COLLECTION_DELAY:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw hand landmarks
            hand_result = hands_detector.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_lm in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            remaining = COLLECTION_DELAY - (time.time() - wait_start)
            cv2.putText(frame, f"WORD: {{word}}  |  Seq {{seq_idx+1}}/{{SEQUENCES_PER_WORD}}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"GET READY... {{remaining:.1f}}s",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Collect Data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Record frames
        frame_idx = 0
        while frame_idx < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats = extract_features(rgb, include_face=True)
            sequence_frames.append(feats)

            # Draw hand landmarks
            hand_result = hands_detector.process(rgb)
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
            cv2.putText(frame, f"RECORDING: {{word}}  [{{frame_idx+1}}/{{SEQUENCE_LENGTH}}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Collect Data", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

        # Save sequence
        if len(sequence_frames) == SEQUENCE_LENGTH:
            seq_array = np.array(sequence_frames)
            save_path = os.path.join(DATA_DIR, word, f"{{seq_idx}}.npy")
            np.save(save_path, seq_array)
            print(f"Saved sequence {{seq_idx}} → {{save_path}}")
            seq_idx += 1
        else:
            print("Incomplete sequence skipped.")

        if cv2.waitKey(1000) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Data collection complete for '{{word}}'")

collect_for_word("{word}")
"""

    # Write and run temporary script
    with open("temp_collect.py", "w") as f:
        f.write(temp_script)

    try:
        subprocess.run([sys.executable, "temp_collect.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Data collection failed!")
    finally:
        # Clean up temp file
        if os.path.exists("temp_collect.py"):
            os.remove("temp_collect.py")

def retrain_model():
    """Retrain the model with current vocabulary."""
    vocab = load_vocabulary()

    if not vocab["words"]:
        print("❌ No words in vocabulary! Add some words first.")
        return

    print(f"🔄 Retraining model with {len(vocab['words'])} words...")
    print("This may take a few minutes...")

    # Import training dependencies
    import subprocess
    import sys

    try:
        # Run the training script
        result = subprocess.run([sys.executable, "train_model.py"],
                              capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("✅ Model retrained successfully!")
            print("📁 New model saved to model/ directory")
        else:
            print("❌ Training failed!")
            print("Error output:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 5 minutes!")
    except FileNotFoundError:
        print("❌ train_model.py not found!")

def show_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("🎯 DYNAMIC SIGN LANGUAGE TRAINER")
    print("="*60)
    print("1. 📝 Add new word")
    print("2. 🗑️  Remove word")
    print("3. 🎥 Collect data for word")
    print("4. 🔄 Retrain model")
    print("5. 📚 List current words")
    print("6. 🚪 Exit")
    print("="*60)

def main():
    print("🎯 Welcome to Dynamic Sign Language Trainer!")

    while True:
        show_menu()
        try:
            choice = input("Choose an option (1-6): ").strip()

            if choice == "1":
                word = input("Enter word to add: ").strip()
                if word:
                    add_word(word)
                else:
                    print("❌ Word cannot be empty!")

            elif choice == "2":
                list_words()
                word = input("Enter word to remove: ").strip()
                if word:
                    remove_word(word)
                else:
                    print("❌ Word cannot be empty!")

            elif choice == "3":
                list_words()
                word = input("Enter word to collect data for: ").strip()
                if word:
                    collect_word_data(word)
                else:
                    print("❌ Word cannot be empty!")

            elif choice == "4":
                retrain_model()

            elif choice == "5":
                list_words()

            elif choice == "6":
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice! Please enter 1-6.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()