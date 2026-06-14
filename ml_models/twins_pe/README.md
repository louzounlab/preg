# twins_pe (vendored)

This directory was **vendored** into `modelsSite` from its standalone repository
so the whole site deploys with a single `git clone` (it used to be a git
submodule, which broke deployment because no `.gitmodules` was committed).

It powers the **PE Twins** and **GDM** prediction pages, via
[`ml_models/adapters/twin_pe.py`](../adapters/twin_pe.py) and
[`ml_models/adapters/gdm.py`](../adapters/gdm.py).

- **Upstream (canonical source):** https://github.com/louzounlab/twins_pe
- **Vendored at upstream commit:** `cc1f0e5` ("gdm")

To pull upstream changes, fetch them from the upstream repo and copy the updated
files in. Do **not** re-add this as a submodule — the site relies on these files
being committed directly in this repo.
