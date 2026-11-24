"""Collect everything needed to run and review the project into one shareable, GitHub-ready folder.
Files over GitHub's 100 MB limit (cache/sim_full/visible.npz) are stored as 45 MB parts.

python tools/make_submission.py [out_dir]   (default: submission/ECE8803_FinalProject_SelectiveBurnIn)
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from burnin.config import split_file

NAME = "ECE8803_FinalProject_SelectiveBurnIn"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission" / NAME

COPY = [
    ("SUBMISSION_README.md", "README.md"),
    ("requirements.txt", "requirements.txt"),
    ("submission_gitignore.txt", ".gitignore"),
    ("src/burnin", "src/burnin"),
    ("notebooks", "notebooks"),
    ("tools/build_notebooks.py", "tools/build_notebooks.py"),
    ("tools/fill_deck.py", "tools/fill_deck.py"),
    ("tools/make_submission.py", "tools/make_submission.py"),
    ("tools/split_large_files.py", "tools/split_large_files.py"),
    ("tools/slide_figures.py", "tools/slide_figures.py"),
    ("cache/models/encoder_wm811k.pt", "cache/models/encoder_wm811k.pt"),
    ("cache/models/encoder_wm811k.json", "cache/models/encoder_wm811k.json"),
    ("cache/models/unet_pretrained_sim_full.pt", "cache/models/unet_pretrained_sim_full.pt"),
    ("cache/models/unet_scratch_sim_full.pt", "cache/models/unet_scratch_sim_full.pt"),
    ("cache/sim_dev", "cache/sim_dev"),
    ("cache/sim_full/wm811k_split.npz", "cache/sim_full/wm811k_split.npz"),
    ("cache/sim_full/hidden.npz", "cache/sim_full/hidden.npz"),
    ("WM811K Dataset/wm811k_dev.npz", "WM811K Dataset/wm811k_dev.npz"),
    ("results/sim_full", "results/sim_full"),
    ("BurnIn_Project_Brief 1.md", "docs/BurnIn_Project_Brief.md"),
    ("Simulator_Outputs_and_Model_Inputs.md", "docs/Simulator_Outputs_and_Model_Inputs.md"),
    ("BurnIn_Presentation_v3.pptx", "docs/BurnIn_Presentation_v3.pptx"),
    ("BurnIn_Presentation_v3.pdf", "docs/BurnIn_Presentation_v3.pdf"),
]
SKIP_NAMES = {"__pycache__", ".ipynb_checkpoints", "oof_baselines.npz", "epochs20"}


def ignore(_, names):
    return [n for n in names if n in SKIP_NAMES]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    missing = []
    for src, dst in COPY:
        s, d = ROOT / src, OUT / dst
        if not s.exists():
            missing.append(src)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, ignore=ignore)
        else:
            shutil.copy2(s, d)
    big = ROOT / "cache" / "sim_full" / "visible.npz"
    if big.exists():
        dst = OUT / "cache" / "sim_full" / "visible.npz"
        shutil.copy2(big, dst)
        split_file(dst)
        dst.unlink()
    for f in OUT.rglob("*"):
        if f.is_file() and f.stat().st_size > 95 * 2**20:
            print(f"WARNING: {f.relative_to(OUT)} is {f.stat().st_size / 2**20:.0f} MB (GitHub limit 100 MB)")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT} ({size / 2**20:.1f} MB)")
    if missing:
        print("missing (not copied):", ", ".join(missing))


if __name__ == "__main__":
    main()
