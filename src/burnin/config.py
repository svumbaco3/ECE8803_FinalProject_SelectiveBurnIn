import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GRID_H = 193
GRID_W = 115
PARAMS = [f"parameter_{i}" for i in range(1, 6)]
TRAIN_LOTS = ["Lot_1", "Lot_2", "Lot_3", "Lot_4"]
TEST_LOT = "Lot_5"
WM811K_CLASSES = ["none", "Edge-Ring", "Edge-Loc", "Center", "Loc", "Scratch", "Random", "Donut", "Near-full"]


def find_project_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "burnin").is_dir() and (q / "cache").is_dir():
            return q
    raise FileNotFoundError("project root (folder containing src/burnin and cache/) not found")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def split_file(path, chunk_mb=45):
    """Split `path` into `<name>.part000, .part001, ...` plus `<name>.parts.json` (size + SHA-256)."""
    path = Path(path)
    chunk = chunk_mb << 20
    for old in path.parent.glob(path.name + ".part*"):
        old.unlink()
    parts = []
    with open(path, "rb") as f:
        for i, block in enumerate(iter(lambda: f.read(chunk), b"")):
            p = path.with_name(f"{path.name}.part{i:03d}")
            p.write_bytes(block)
            parts.append(p.name)
    manifest = {"file": path.name, "size": path.stat().st_size, "sha256": _sha256(path), "parts": parts}
    path.with_name(path.name + ".parts.json").write_text(json.dumps(manifest, indent=2))
    return parts


def ensure_joined(path):
    """If `path` is missing but its split parts exist, reassemble and verify it. Returns `path`."""
    path = Path(path)
    man = path.with_name(path.name + ".parts.json")
    if path.exists() or not man.exists():
        return path
    m = json.loads(man.read_text())
    tmp = path.with_name(path.name + ".joining")
    with open(tmp, "wb") as out:
        for name in m["parts"]:
            out.write((path.parent / name).read_bytes())
    if tmp.stat().st_size != m["size"] or _sha256(tmp) != m["sha256"]:
        tmp.unlink()
        raise IOError(f"reassembled {path.name} does not match {man.name}; re-download the parts")
    tmp.replace(path)
    print(f"reassembled {path.name} from {len(m['parts'])} parts")
    return path


@dataclass
class Paths:
    root: Path
    sim: str = "sim_dev"

    @property
    def sim_dir(self):
        return self.root / "cache" / self.sim

    @property
    def visible(self):
        return ensure_joined(self.sim_dir / "visible.npz")

    @property
    def hidden(self):
        return ensure_joined(self.sim_dir / "hidden.npz")

    @property
    def wm811k_split(self):
        p = self.sim_dir / "wm811k_split.npz"
        return p if p.exists() else self.root / "cache" / "sim_dev" / "wm811k_split.npz"

    @property
    def wm811k(self):
        return self.root / "WM811K Dataset" / "wm811k_dev.npz"

    @property
    def models(self):
        d = self.root / "cache" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results(self):
        d = self.root / "results" / self.sim
        d.mkdir(parents=True, exist_ok=True)
        return d
