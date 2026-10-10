"""Minimal notebook executor used to build the .ipynb with real outputs.

Jupyter is not required to *read* the notebook - this helper only exists so the
committed notebook ships with its outputs already populated.
Usage:  python src/run_notebook.py
"""
import ast, base64, io, json, sys, time, traceback, contextlib, uuid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from nb_cells import CELLS  # noqa: E402

ns = {"__name__": "__main__"}
exec_count = 0
nb_cells = []


def to_lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def grab_figures():
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        outs.append({
            "output_type": "display_data",
            "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                     "text/plain": ["<Figure>"]},
            "metadata": {},
        })
        plt.close(fig)
    return outs


plt.show = lambda *a, **k: None  # figures are harvested after each cell

for kind, src in CELLS:
    cid = uuid.uuid4().hex[:8]
    if kind == "md":
        nb_cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": to_lines(src)})
        continue

    exec_count += 1
    outputs = []
    buf = io.StringIO()
    t0 = time.time()
    try:
        tree = ast.parse(src)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, "<cell>", "exec"), ns)
            result = eval(compile(last, "<cell>", "eval"), ns) if last is not None else None
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": to_lines(buf.getvalue())})
        outputs.extend(grab_figures())
        if result is not None:
            data = {"text/plain": to_lines(repr(result))}
            if hasattr(result, "_repr_html_"):
                data["text/html"] = to_lines(result._repr_html_())
            outputs.append({"output_type": "execute_result", "execution_count": exec_count,
                            "data": data, "metadata": {}})
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        outputs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback", "traceback": tb.split("\n")})
        nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": exec_count,
                         "outputs": outputs, "source": to_lines(src)})
        break
    print(f"[cell {exec_count}] {time.time() - t0:.1f}s", file=sys.stderr)
    nb_cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": exec_count,
                     "outputs": outputs, "source": to_lines(src)})

nb = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open("Risk_Alert_Classifier.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("notebook written:", len(nb_cells), "cells", file=sys.stderr)
