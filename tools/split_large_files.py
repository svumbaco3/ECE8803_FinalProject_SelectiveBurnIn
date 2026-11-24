"""Split files that exceed GitHub's 100 MB limit into 45 MB parts, or reassemble them.

python tools/split_large_files.py split cache/sim_full/visible.npz [more files...]
python tools/split_large_files.py join  cache/sim_full/visible.npz
The notebooks reassemble automatically (burnin.config.ensure_joined), so `join` is optional.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from burnin.config import ensure_joined, split_file

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("split", "join"):
        sys.exit(__doc__)
    for f in sys.argv[2:]:
        if sys.argv[1] == "split":
            print(f, "->", ", ".join(split_file(f)))
        else:
            ensure_joined(f)
