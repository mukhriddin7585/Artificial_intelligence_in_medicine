from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for folder in sorted(root.glob("[0-9][0-9]_*")):
    print(f"\n=== {folder.name} ===", flush=True)
    subprocess.run([sys.executable, "main.py"], cwd=folder, check=True)
print("15 loyiha bajarildi")
