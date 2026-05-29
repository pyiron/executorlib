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
