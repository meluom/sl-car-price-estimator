"""Convert a # %%-marked .py source into a Jupyter notebook.
Usage: python build_notebook.py [src.py out.ipynb]"""
import re, sys, nbformat
SRC, OUT = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("notebook_src.py", "car_price_riyasewana_2026.ipynb")
src = open(SRC).read()
cells = []
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    if "[markdown]" in header:
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbformat.write(nb, OUT)
print(len(cells), "cells")
