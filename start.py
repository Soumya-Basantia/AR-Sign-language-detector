"""
start.py — Unified Launcher for AR Sign Language Communication Platform
========================================================================
Interactive terminal launcher and CLI entrypoint for all subsystems:
  1. Glassmorphic Web Dashboard & Two-Way Dialogue Bridge (FastAPI + Browser)
  2. OpenCV Desktop AR Glasses HUD (Local Real-Time Gesture Inference)
  3. Automated System Verification Tests
  4. Baseline Model Generator / Training Studio
  5. Custom Sequence & Letter Data Collector
"""

import sys
import os
import subprocess
import webbrowser
import time
import argparse


BANNER = r"""
======================================================================
     _    ____    ____  _             _                               
    / \  |  _ \  / ___|(_) __ _ _ __ | |    __ _ _ __   __ _         
   / _ \ | |_) | \___ \| |/ _` | '_ \| |   / _` | '_ \ / _` |        
  / ___ \|  _ <   ___) | | (_| | | | | |__| (_| | | | | (_| |        
 /_/   \_\_| \_\ |____/|_|\__, |_| |_|_____\__,_|_| |_|\__, |        
                          |___/                         |___/         
        Real-Time AR Assistive Communication System v2.0
======================================================================
"""


def check_dependencies():
    """Verify that required packages are installed."""
    required = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "requests": "requests",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)
    return missing


def launch_web(port=8000, host="127.0.0.1"):
    """Launch the FastAPI glassmorphic web dashboard and open browser."""
    print(f"\n[INFO] Starting Glassmorphic Web Dashboard on http://{host}:{port}...")
    url = f"http://{host}:{port}"
    
    # Open browser after a short delay
    def _open():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
        from web_app import app
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("\n[INFO] Web dashboard stopped.")


def launch_desktop():
    """Launch the OpenCV AR Glasses desktop application."""
    print("\n[INFO] Starting OpenCV Desktop AR HUD Application...")
    print("[INFO] Controls: SPACE=Accept, ENTER=Speak, M=Mode, G=AR HUD, Q=Quit")
    script = os.path.join(os.path.dirname(__file__), "predict_sequence.py")
    subprocess.run([sys.executable, script])


def run_tests():
    """Run the 14-test verification suite."""
    print("\n[INFO] Running Automated Verification Test Suite...")
    script = os.path.join(os.path.dirname(__file__), "test_system.py")
    subprocess.run([sys.executable, "-m", "unittest", script, "-v"])


def generate_models():
    """Generate or retrain the baseline gesture & letter models."""
    print("\n[INFO] Provisioning Baseline Gesture & Letter Models...")
    from generate_baseline_model import train_and_save_baseline_models
    train_and_save_baseline_models()


def launch_collector():
    """Launch the sequence collector tool."""
    print("\n[INFO] Select Collector Mode:")
    print("  [1] Word Gesture Sequences (collect_sequences.py)")
    print("  [2] Fingerspelling Letters (collect_letters.py)")
    choice = input("Select [1/2]: ").strip()
    if choice == "1":
        script = os.path.join(os.path.dirname(__file__), "collect_sequences.py")
        subprocess.run([sys.executable, script])
    elif choice == "2":
        script = os.path.join(os.path.dirname(__file__), "collect_letters.py")
        subprocess.run([sys.executable, script])
    else:
        print("[WARN] Invalid selection.")


def interactive_menu():
    """Interactive console menu."""
    while True:
        print(BANNER)
        print("Select an option to launch:")
        print("  [1] Launch Web Dashboard (Browser UI + AR Viewport + Two-Way Speech)")
        print("  [2] Launch Desktop AR HUD (OpenCV Camera Overlay + Holographic Glasses)")
        print("  [3] Run Automated System Tests (14 Unit Tests)")
        print("  [4] Provision / Retrain Baseline Models (16 Words + 26 Letters)")
        print("  [5] Collect New Gesture / Letter Training Data")
        print("  [Q] Exit")
        print("-" * 70)
        
        choice = input("Enter choice [1-5 or Q]: ").strip().lower()
        if choice in ["1", "web"]:
            launch_web()
            break
        elif choice in ["2", "desktop", "cv"]:
            launch_desktop()
            break
        elif choice in ["3", "test", "tests"]:
            run_tests()
            input("\nPress Enter to return to menu...")
        elif choice in ["4", "model", "baseline"]:
            generate_models()
            input("\nPress Enter to return to menu...")
        elif choice in ["5", "collect"]:
            launch_collector()
            input("\nPress Enter to return to menu...")
        elif choice in ["q", "exit", "quit"]:
            print("\n[INFO] Exiting. Goodbye!")
            sys.exit(0)
        else:
            print("\n[ERROR] Unrecognized selection. Please try again.")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="AR Sign Language System Unified Launcher")
    parser.add_argument("--web", action="store_true", help="Launch the Web Dashboard immediately")
    parser.add_argument("--desktop", action="store_true", help="Launch the OpenCV Desktop AR HUD immediately")
    parser.add_argument("--test", action="store_true", help="Run automated unit test suite")
    parser.add_argument("--models", action="store_true", help="Provision baseline models")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")
    args = parser.parse_args()

    missing = check_dependencies()
    if missing:
        print(BANNER)
        print(f"[WARN] Missing dependencies: {', '.join(missing)}")
        print("[INFO] Run: pip install " + " ".join(missing))
        resp = input("Would you like to attempt installing them now? [y/N]: ").strip().lower()
        if resp == "y":
            subprocess.run([sys.executable, "-m", "pip", "install", *missing])
        else:
            print("[WARN] Continuing with available modules...")

    if args.web:
        launch_web(port=args.port)
    elif args.desktop:
        launch_desktop()
    elif args.test:
        run_tests()
    elif args.models:
        generate_models()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
