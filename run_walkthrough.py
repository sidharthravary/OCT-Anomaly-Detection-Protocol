"""Run notebooks/project_walkthrough.ipynb cell by cell, printing each section's output live.

Saves the executed notebook and an HTML copy (outputs/walkthrough.html).
Usage (project root):  .venv-gpu\Scripts\python.exe run_walkthrough.py
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path("notebooks/project_walkthrough.ipynb")
nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
t0 = time.time()
with client.setup_kernel():
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            head = cell.source.strip().splitlines()[0]
            if head.startswith("#"):
                print("\n" + "=" * 90 + f"\n{head.lstrip('# ')}\n" + "=" * 90, flush=True)
            continue
        client.execute_cell(cell, i)
        for o in cell.outputs:
            if o.output_type == "stream":
                print(o.text.rstrip(), flush=True)
            elif o.output_type == "error":
                print(f"ERROR {o.ename}: {o.evalue}", flush=True)
            elif "data" in o:
                d = o["data"]
                if "image/png" in d:
                    print("   [figure]", flush=True)
                elif "text/plain" in d and "IPython.core.display" not in d["text/plain"]:
                    print(d["text/plain"], flush=True)
nbformat.write(nb, NB)
try:
    from nbconvert import HTMLExporter
    html, _ = HTMLExporter().from_notebook_node(nb)
    Path("outputs/walkthrough.html").write_text(html, encoding="utf-8")
    print(f"\nHTML -> outputs/walkthrough.html")
except ImportError:
    pass
print(f"\nDONE: all {sum(c.cell_type == 'code' for c in nb.cells)} code cells ran in {time.time() - t0:.0f}s")
