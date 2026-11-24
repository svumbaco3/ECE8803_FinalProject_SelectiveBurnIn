import argparse
import re
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            body = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            cells.append(new_markdown_cell(body))
        elif body.strip():
            cells.append(new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(name, execute, timeout):
    src = SRC / f"{name}.py"
    nb = new_notebook(cells=parse(src.read_text(encoding="utf-8")))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    if execute:
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
        try:
            NotebookClient(nb, timeout=timeout, kernel_name="python3",
                           resources={"metadata": {"path": str(OUT)}}).execute()
        except CellExecutionError:
            nbformat.write(nb, OUT / f"{name}.ipynb")
            raise
    out = OUT / f"{name}.ipynb"
    nbformat.write(nb, out)
    print("wrote", out.name, "(executed)" if execute else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    for n in a.names or sorted(p.stem for p in SRC.glob("*.py")):
        build(n, a.execute, a.timeout)
