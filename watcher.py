import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------------- CONFIG ----------------
# The specific file we are watching
WATCH_FILE = "DATASET.csv"
# The script to run when the file changes
TRAIN_SCRIPT = "trainlinear.py"

class TrainHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Check if the modified file is actually DATASET.csv
        if event.src_path.endswith(WATCH_FILE):
            print(f"\n👀 Detected change in: {WATCH_FILE}")
            print("🚀 Triggering model retraining...")
            
            # Run the training script
            subprocess.run(["python", TRAIN_SCRIPT], shell=True)
            
            print("✅ Retraining complete. Waiting for next update...")

if __name__ == "__main__":
    # Get the current folder where this script is running
    path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(path, "data")
    
    print(f"🕵️‍♂️ WATCHER STARTED")
    print(f"📂 Watching folder: {data_path}")
    print(f"🎯 Target file: {WATCH_FILE}")
    print("------------------------------------------------")

    event_handler = TrainHandler()
    observer = Observer()
    
    # We watch the 'data' folder specifically
    observer.schedule(event_handler, path=data_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()