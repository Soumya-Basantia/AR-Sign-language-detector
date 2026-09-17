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
    """Collect data for a specific word directly using webcam."""
    vocab = load_vocabulary()

    if word not in vocab["words"]:
        print(f"❌ Word '{word}' not in vocabulary! Add it first.")
        return

    print(f"🎥 Starting data collection for: {word}")
    print("Make sure your webcam is ready...")

    from collect_sequences import collect_for_word
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam! Check camera connection.")
        return

    try:
        collect_for_word(word, cap)
        print(f"✅ Data collection complete for '{word}'")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

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