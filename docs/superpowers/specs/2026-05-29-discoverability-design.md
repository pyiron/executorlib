# Design: Improve executorlib discoverability for scientist end users

**Date:** 2026-05-29
**Status:** Approved (design phase)

## Goal

Make the executorlib repository more attractive to **scientist end users**, with
the specific bottleneck being **discovery** — people who would benefit never hear
about executorlib, or land on it and cannot quickly tell what it is or when to use
it. This design covers **concrete repo/docs changes** (not a broader outreach
strategy doc).

Chosen approach: **B — Findability essentials + a positioning page.** Pair cheap,
high-leverage metadata wins with the one content asset (comparison/positioning)
that most directly converts "when-to-use" searches into inbound traffic.

## Current state (baseline)

- 73 stars, 6 forks, 6 watchers.
- Strong existing presentation: JOSS paper, ReadTheDocs (jupyter-book), Binder,
  codecov, CI badges. README progresses single-node → cluster → job.
- GitHub topics: `hpc, mpi, mpi4py, multiprocessing, pyiron, flux, slurm`.
- **No custom social-preview image** (uses GitHub's auto-generated card).
- README has **no positioning / comparison** to alternatives.
- No discoverability-focused docs page; comparison search queries go uncaptured.
- `pyproject.toml` keywords: `high performance computing, hpc, task scheduler,
  slurm, flux-framework, executor`.

## Components

Split by ownership: repo-file changes (implemented in this work) vs GitHub web-UI
settings (Jan performs; this work supplies exact values/assets) vs outreach
(checklist supplied; Jan executes).

### 1. README "Why executorlib?" section (repo file)

Insert a new section near the top of `README.md`, after **Key Features** and before
**Examples**, containing:

- A short **positioning thesis** paragraph (see below).
- A **condensed comparison table** (the dimensions below, abbreviated cells).
- A link to the full `docs/comparison.md` page.

### 2. Positioning docs page (repo file)

New `docs/comparison.md`, registered in `docs/_toc.yml` (placed after
`installation.md`, before `1-single-node.ipynb`, so it is high in the nav).

Title/H1 chosen for search: **"executorlib vs Dask, Parsl, Ray & Snakemake —
when to use which"**.

Content:
- Intro restating the positioning thesis.
- The full comparison table (dimensions below).
- One short, **fair** paragraph per alternative, each with a link to that tool and
  an honest **"use \<tool\> instead when…"** note.
- A closing "choose executorlib when…" summary.

### 3. PyPI / repo keyword tune (repo file)

Extend `keywords` in `pyproject.toml` with discovery terms not already present,
e.g.: `python`, `parallel-computing`, `distributed-computing`,
`scientific-computing`, `concurrent-futures`, `mpi4py`, `hpc-cluster`, `gpu`.
(Final list curated to avoid duplicates with existing keywords.)

### 4. Expanded GitHub topics (GitHub setting — Jan applies)

Supply a final topic list to paste into the repo's topics. Candidate additions to
the existing set: `python`, `parallel-computing`, `distributed-computing`,
`scientific-computing`, `concurrent-futures`, `task-scheduler`, `hpc-cluster`,
`gpu`. (GitHub caps topics at 20; final list trimmed to the highest-traffic terms.)

### 5. Custom social-preview card (asset generated; Jan uploads)

Generate a branded **1280×640 PNG** social-preview card (name + one-line value
prop + visual cue consistent with the pyiron brand). Delivered as a repo file
(e.g. `docs/images/social-preview.png`); Jan uploads it under
**Settings → General → Social preview**.

### 6. Awesome-list submission checklist (outreach — Jan executes)

Provide a checklist of target lists (e.g. awesome-python, awesome HPC /
scientific-computing lists) with suggested one-line entry text. Jan opens the PRs.

## Comparison content specification

**Alternatives covered:** Dask, Parsl, Ray, Snakemake, plus `concurrent.futures`
as the stdlib baseline executorlib extends.

**Comparison dimensions (table columns):**

1. Drop-in `concurrent.futures.Executor` API compatibility
2. Per-function-call resource assignment (cores / GPUs / threads)
3. Native HPC scheduler integration (SLURM, flux)
4. MPI-parallel function support
5. Caching of intermediate results
6. Setup / learning overhead

**Positioning thesis:**

> executorlib is the lightest path to take *existing* Python functions and scale
> them across HPC nodes with per-call resource control and native SLURM/flux
> integration, without rewriting code into a new paradigm. Dask and Ray ask you to
> adopt their data/actor model; Snakemake and Parsl ask you to author workflows.
> executorlib extends the standard-library `Executor` interface you already know.

**Credibility/SEO stance:** each alternative gets an honest "use it instead
when…" caveat. This is a deliberate trade — minor competitive concession in
exchange for trust and search relevance.

## Out of scope

- Logo / brand identity work and docs front-page redesign (these were approach C).
- Any non-repo outreach beyond the awesome-list checklist (talks, blog posts,
  social campaigns).
- Functional/code changes to executorlib itself.

## Success criteria

- README clearly answers "what is this and when do I use it" within the first
  screenful, including a comparison table.
- A search-indexable `docs/comparison.md` exists and is reachable high in the docs
  nav.
- `pyproject.toml` keywords and a supplied GitHub-topics list cover the
  high-traffic discovery terms.
- A branded social-preview PNG exists in the repo, ready to upload.
- A concrete awesome-list submission checklist is delivered.

## Verification

- Build the docs locally (jupyter-book) and confirm `comparison.md` renders and
  appears in the nav.
- Render the README (GitHub markdown) and confirm the table displays and the link
  resolves.
- Confirm `pyproject.toml` still parses (e.g. `python -m build` dry check or a
  lightweight TOML parse).
