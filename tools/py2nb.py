"""Convert a '# %%'-cell script into a Jupyter notebook:  python tools/py2nb.py script.py out.ipynb"""
import json
import sys
from pathlib import Path


def convert(src, dst):
    cells, cur, kind = [], [], "code"

    def flush():
        text = "\n".join(cur).strip("\n")
        if text:
            if kind == "markdown":
                text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            lines = text.splitlines(keepends=True)
            cell = {"cell_type": kind, "metadata": {}, "source": lines}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)

    for line in Path(src).read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            cur, kind = [], "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"},
                                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
          "nbformat": 4, "nbformat_minor": 5}
    Path(dst).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
