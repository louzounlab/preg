# twin_efw (vendored)

This directory was **vendored** into `modelsSite` from its standalone repository
so the whole site deploys with a single `git clone` (it used to be a git
submodule, which broke deployment because no `.gitmodules` was committed).

It powers the **Twin-EFW** (twin estimated fetal weight) page, via
[`ml_models/adapters/twin_efw.py`](../adapters/twin_efw.py).

> Note: the upstream repository is named **Twin-FWE**; the page/module were
> renamed to **EFW** (estimated fetal weight) on this site to fix the acronym.

- **Upstream (canonical source):** https://github.com/louzounlab/Twin-FWE
- **Vendored at upstream commit:** `8e30886` ("Update about.html")

To pull upstream changes, fetch them from the upstream repo and copy the updated
files in. Do **not** re-add this as a submodule — the site relies on these files
being committed directly in this repo.
