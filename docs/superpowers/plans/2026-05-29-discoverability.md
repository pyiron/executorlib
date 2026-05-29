# Discoverability Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make executorlib easier for scientist end users to discover and understand by adding repo positioning content, a search-optimized comparison page, tuned keywords/topics, a branded social-preview card, and a manual-steps checklist.

**Architecture:** Pure documentation/metadata work — no changes to `src/`. A new "Why executorlib?" README section and a `docs/comparison.md` page deliver positioning; `pyproject.toml` keywords and a supplied GitHub-topics list improve search indexing; an SVG→PNG social card improves social-share click-through; a manual-steps file captures the GitHub-UI and awesome-list actions for Jan.

**Tech Stack:** Markdown, jupyter-book (docs), TOML (`pyproject.toml`), SVG + `cairosvg` (social card).

---

## File Structure

- `README.md` — add a "Why executorlib?" section (positioning thesis + condensed table + link).
- `docs/comparison.md` — **new**, full comparison page (positioning, full table, per-tool paragraphs).
- `docs/_toc.yml` — register `comparison.md` in the nav.
- `pyproject.toml` — extend `keywords`.
- `docs/images/social-preview.svg` — **new**, source for the social card.
- `docs/images/social-preview.png` — **new**, generated 1280×640 card for GitHub upload.
- `docs/superpowers/discoverability-manual-steps.md` — **new**, checklist of actions only Jan can perform (GitHub topics, social-preview upload, awesome-list PRs). Intentionally NOT added to `_toc.yml`.

Source of truth for shared content: the **comparison table** and **positioning thesis** are authored in full in `docs/comparison.md`; the README carries a condensed copy. Keep the thesis wording identical between the two.

---

### Task 1: Add "Why executorlib?" section to README

**Files:**
- Modify: `README.md` (insert between line 21 blank line and line 22 `## Examples`)

- [ ] **Step 1: Insert the new section**

Insert the following block immediately before the `## Examples` line in `README.md`:

```markdown
## Why executorlib?
executorlib is the lightest path to take *existing* Python functions and scale them across high performance computing
(HPC) nodes — with per-function-call resource control and native [SLURM](https://slurm.schedmd.com) and
[flux](http://flux-framework.org) integration — without rewriting your code into a new paradigm. It extends the standard
library [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) you already
know, rather than asking you to adopt a new data, actor, or workflow model.

| | executorlib | `concurrent.futures` | [Dask](https://www.dask.org) | [Parsl](https://parsl-project.org) | [Ray](https://www.ray.io) | [Snakemake](https://snakemake.github.io) |
|---|---|---|---|---|---|---|
| Drop-in `Executor` API | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| Per-call resource assignment | ✅ | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| Native HPC scheduler (SLURM/flux) | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| MPI-parallel functions | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| Caching of results | ✅ | ❌ | ⚠️ | ✅ | ❌ | ✅ |
| Setup / learning overhead | Low | Very low | Medium | Medium | Medium | High |

✅ first-class · ⚠️ possible via add-on/config · ❌ not supported. See the full
[comparison: when to use which](https://executorlib.readthedocs.io/en/latest/comparison.html) for honest guidance on
when another tool is the better fit.
```

- [ ] **Step 2: Verify the README renders**

Run: `python -c "import pathlib; t=pathlib.Path('README.md').read_text(); assert '## Why executorlib?' in t and t.index('## Why executorlib?') < t.index('## Examples'); print('OK: section present and ordered')"`
Expected: `OK: section present and ordered`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Why executorlib positioning section to README"
```

---

### Task 2: Create the comparison docs page

**Files:**
- Create: `docs/comparison.md`

- [ ] **Step 1: Write the full comparison page**

Create `docs/comparison.md` with exactly this content:

```markdown
# executorlib vs Dask, Parsl, Ray & Snakemake — when to use which

executorlib is the lightest path to take *existing* Python functions and scale them across high performance computing
(HPC) nodes — with per-function-call resource control and native [SLURM](https://slurm.schedmd.com) and
[flux](http://flux-framework.org) integration — without rewriting your code into a new paradigm. It extends the standard
library [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) you already
know, rather than asking you to adopt a new data, actor, or workflow model.

This page compares executorlib with the tools scientists most often weigh it against, and is honest about when each
alternative is the better choice.

## At a glance

| | executorlib | `concurrent.futures` | Dask | Parsl | Ray | Snakemake |
|---|---|---|---|---|---|---|
| Drop-in `Executor` API | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| Per-call resource assignment | ✅ | ❌ | ⚠️ | ✅ | ✅ | ✅ |
| Native HPC scheduler (SLURM/flux) | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| MPI-parallel functions | ✅ | ❌ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| Caching of results | ✅ | ❌ | ⚠️ | ✅ | ❌ | ✅ |
| Setup / learning overhead | Low | Very low | Medium | Medium | Medium | High |

✅ first-class · ⚠️ possible via an add-on or extra configuration · ❌ not supported.

## `concurrent.futures` (the Python standard library)

The [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) module is where most parallel
Python starts: `ProcessPoolExecutor` and `ThreadPoolExecutor` run functions in parallel on a single machine. executorlib
deliberately mirrors this `Executor` interface so the step up to HPC is minimal.

**Use `concurrent.futures` instead when** your work fits comfortably on one machine and you do not need HPC schedulers,
per-call resource control, MPI, or caching.

## [Dask](https://www.dask.org)

Dask scales NumPy/pandas-style workloads with parallel arrays, dataframes, and a `delayed`/futures API, and reaches HPC
via [dask-jobqueue](https://jobqueue.dask.org). It is excellent for large out-of-core data structures, but its futures
API is its own, and per-task resources and MPI rely on add-ons.

**Use Dask instead when** your problem is fundamentally about large arrays/dataframes or out-of-core data, rather than
scheduling independent Python functions across an HPC allocation.

## [Parsl](https://parsl-project.org)

Parsl is the closest conceptual neighbor: a parallel scripting library with strong HPC support, MPI apps, and app-level
caching. It uses its own decorator/app model (`@python_app`) and an executor-configuration layer rather than the standard
library `Executor` interface.

**Use Parsl instead when** you are authoring a larger dataflow of apps and want its app/configuration model, or you need
a provider it supports that executorlib does not.

## [Ray](https://www.ray.io)

Ray is a distributed framework built around remote tasks and stateful actors, widely used for AI/ML and reinforcement
learning. It assigns CPUs/GPUs per task, but adopting Ray means adopting its `@ray.remote` programming model, and HPC
scheduler integration is via cluster launchers rather than native SLURM/flux.

**Use Ray instead when** you need long-lived stateful actors, an AI/ML ecosystem, or a distributed-object model — and you
are willing to write code in Ray's paradigm.

## [Snakemake](https://snakemake.github.io)

Snakemake is a file-oriented workflow manager: you declare rules with inputs/outputs and it builds a dependency graph,
with strong HPC support and file-based caching. It is a workflow DSL, not a drop-in way to parallelize Python functions.

**Use Snakemake instead when** your pipeline is naturally expressed as files transformed by rules, and you want
reproducible, file-driven workflow management rather than in-process Python futures.

## Choose executorlib when

- You already have Python functions and want to scale them across HPC nodes with minimal rewriting.
- You want to assign cores, threads, or GPUs **per function call**.
- You want native [SLURM](https://slurm.schedmd.com) / [flux](http://flux-framework.org) integration and optional MPI
  parallelism inside your functions.
- You want optional caching of intermediate results for rapid, iterative prototyping in notebooks.
```

- [ ] **Step 2: Verify the page parses and references the right tools**

Run: `python -c "import pathlib; t=pathlib.Path('docs/comparison.md').read_text(); assert all(k in t for k in ['# executorlib vs', 'Dask','Parsl','Ray','Snakemake','Choose executorlib when']); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add docs/comparison.md
git commit -m "docs: add comparison page positioning executorlib vs alternatives"
```

---

### Task 3: Register the comparison page in the docs nav

**Files:**
- Modify: `docs/_toc.yml`

- [ ] **Step 1: Add the entry**

In `docs/_toc.yml`, change:

```yaml
chapters:
- file: installation.md
- file: 1-single-node.ipynb
```

to:

```yaml
chapters:
- file: installation.md
- file: comparison.md
- file: 1-single-node.ipynb
```

- [ ] **Step 2: Verify the TOC is valid YAML and contains the entry**

Run: `python -c "import yaml; d=yaml.safe_load(open('docs/_toc.yml')); files=[c.get('file') for c in d['chapters']]; assert 'comparison.md' in files and files.index('comparison.md')==files.index('installation.md')+1; print('OK:', files)"`
Expected: `OK:` followed by the chapter list with `comparison.md` right after `installation.md`

- [ ] **Step 3: Commit**

```bash
git add docs/_toc.yml
git commit -m "docs: add comparison page to documentation nav"
```

---

### Task 4: Extend PyPI keywords

**Files:**
- Modify: `pyproject.toml` (the `keywords = [...]` line under `[project]`)

- [ ] **Step 1: Replace the keywords list**

In `pyproject.toml`, replace:

```toml
keywords = ["high performance computing", "hpc", "task scheduler", "slurm", "flux-framework", "executor"]
```

with:

```toml
keywords = ["high performance computing", "hpc", "task scheduler", "slurm", "flux-framework", "executor", "python", "parallel computing", "distributed computing", "scientific computing", "concurrent.futures", "mpi4py", "gpu"]
```

- [ ] **Step 2: Verify the file still parses and contains the new keywords**

Run: `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); k=d['project']['keywords']; assert 'parallel computing' in k and 'scientific computing' in k; print('OK:', len(k), 'keywords')"`
Expected: `OK: 13 keywords`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: broaden PyPI keywords for discoverability"
```

---

### Task 5: Create the social-preview card

**Files:**
- Create: `docs/images/social-preview.svg`
- Create: `docs/images/social-preview.png`

- [ ] **Step 1: Write the SVG source**

Create `docs/images/social-preview.svg` with exactly this content:

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0b3d61"/>
      <stop offset="1" stop-color="#114e7c"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="640" fill="url(#bg)"/>
  <text x="100" y="250" font-family="Helvetica, Arial, sans-serif" font-size="110" font-weight="700" fill="#ffffff">executorlib</text>
  <text x="100" y="340" font-family="Helvetica, Arial, sans-serif" font-size="46" font-weight="400" fill="#cfe3f3">Up-scale your Python functions for HPC</text>
  <text x="100" y="430" font-family="Helvetica, Arial, sans-serif" font-size="34" font-weight="400" fill="#9ec7e6">Per-call CPU/GPU resources · native SLURM &amp; flux · MPI · caching</text>
  <text x="100" y="560" font-family="Helvetica, Arial, sans-serif" font-size="30" font-weight="400" fill="#7fb2d8">github.com/pyiron/executorlib</text>
</svg>
```

- [ ] **Step 2: Convert the SVG to a 1280×640 PNG**

Run: `pip install cairosvg && python -c "import cairosvg; cairosvg.svg2png(url='docs/images/social-preview.svg', write_to='docs/images/social-preview.png', output_width=1280, output_height=640)"`
Expected: command exits 0, no errors. (If `cairosvg` is unavailable in the environment, fall back to: `rsvg-convert -w 1280 -h 640 docs/images/social-preview.svg -o docs/images/social-preview.png`.)

- [ ] **Step 3: Verify the PNG dimensions**

Run: `python -c "import struct; d=open('docs/images/social-preview.png','rb').read(); assert d[:8]==b'\x89PNG\r\n\x1a\n'; w,h=struct.unpack('>II', d[16:24]); assert (w,h)==(1280,640); print('OK:', w, 'x', h)"`
Expected: `OK: 1280 x 640`

- [ ] **Step 4: Commit**

```bash
git add docs/images/social-preview.svg docs/images/social-preview.png
git commit -m "docs: add branded social-preview card for repository"
```

---

### Task 6: Create the manual-steps checklist

**Files:**
- Create: `docs/superpowers/discoverability-manual-steps.md`

This file captures the actions that can only be performed in the GitHub web UI or in third-party repos. It is intentionally NOT added to `docs/_toc.yml` (not part of the published site).

- [ ] **Step 1: Write the checklist**

Create `docs/superpowers/discoverability-manual-steps.md` with exactly this content:

```markdown
# Discoverability — manual steps for the maintainer

These actions cannot be done from the repo files alone. Do them in the GitHub web UI / external repos.

## 1. Expand GitHub topics

Settings landing page → ⚙ next to **About** → Topics. Current topics:
`hpc, mpi, mpi4py, multiprocessing, pyiron, flux, slurm`

Add (GitHub allows up to 20):
`python`, `parallel-computing`, `distributed-computing`, `scientific-computing`,
`concurrent-futures`, `task-scheduler`, `hpc-cluster`, `gpu`

## 2. Upload the social-preview card

Settings → General → **Social preview** → upload `docs/images/social-preview.png`.
This is the card shown when the repo link is shared on Slack, X/Twitter, LinkedIn, etc.

## 3. Submit to awesome-lists

Open a PR adding executorlib to each list below. Suggested entry text:

> **[executorlib](https://github.com/pyiron/executorlib)** — Up-scale Python functions for HPC: extends the standard
> `concurrent.futures` Executor with per-call CPU/GPU resources and native SLURM/flux integration.

Target lists (verify each is still active and accepts submissions before submitting):
- [ ] awesome-python (Concurrency / Distributed Computing section)
- [ ] an awesome HPC list (e.g. "awesome-hpc")
- [ ] an awesome scientific-computing / research-software list
- [ ] the pyiron ecosystem README / docs, if not already cross-linked

## 4. (Optional) Enable GitHub Discussions

Settings → General → Features → **Discussions**. Gives users a Q&A channel that is itself indexed and discoverable.
```

- [ ] **Step 2: Verify the file exists and lists the four manual steps**

Run: `python -c "import pathlib; t=pathlib.Path('docs/superpowers/discoverability-manual-steps.md').read_text(); assert all(s in t for s in ['Expand GitHub topics','social-preview card','awesome-lists','Discussions']); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Confirm it is NOT in the published nav**

Run: `python -c "import yaml; d=yaml.safe_load(open('docs/_toc.yml')); assert all('discoverability-manual-steps' not in str(c) for c in d['chapters']); print('OK: excluded from site')"`
Expected: `OK: excluded from site`

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/discoverability-manual-steps.md
git commit -m "docs: add maintainer checklist for GitHub-UI discoverability steps"
```

---

### Task 7: Final docs build verification

**Files:** none (verification only)

- [ ] **Step 1: Build the documentation**

Run: `jupyter-book build docs 2>&1 | tail -20`
Expected: build completes; output reports `comparison.html` was generated and shows no errors for `comparison.md`. (If `jupyter-book` is not installed, install with `pip install jupyter-book` first, or skip and rely on the per-file markdown checks from Tasks 2–3.)

- [ ] **Step 2: Confirm the comparison page was rendered**

Run: `test -f docs/_build/html/comparison.html && echo "OK: comparison.html built"`
Expected: `OK: comparison.html built`

- [ ] **Step 3: No commit needed**

The `docs/_build/` directory is build output (git-ignored). Nothing to commit.

---

## Self-Review

**Spec coverage:**
- README "Why executorlib?" section → Task 1 ✅
- Positioning docs page `docs/comparison.md` + `_toc.yml` → Tasks 2, 3 ✅
- PyPI keyword tune → Task 4 ✅
- Expanded GitHub topics (values supplied) → Task 6 (manual-steps file) ✅
- Custom social-preview card asset → Task 5 ✅
- Awesome-list submission checklist → Task 6 ✅
- Verification (docs build, README/TOML parse) → per-task checks + Task 7 ✅

**Placeholder scan:** No TBD/TODO in the deliverable content. (The awesome-list checklist contains intentional `- [ ]` items for Jan to tick off when submitting PRs — these are deliberate manual actions, not plan placeholders.)

**Type/name consistency:** Tool names (Dask, Parsl, Ray, Snakemake, `concurrent.futures`), the table columns, and the positioning thesis are identical between Task 1 (README condensed) and Task 2 (full page). File paths are consistent across tasks.
