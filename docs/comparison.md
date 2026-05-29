# Comparison

executorlib is the lightest path to take *existing* Python functions and scale them across high performance computing
(HPC) nodes — with per-function-call resource control and native [SLURM](https://slurm.schedmd.com) and
[flux](http://flux-framework.org) integration — without rewriting your code into a new paradigm. It extends the standard
library [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) you already
know, rather than asking you to adopt a new data, actor, or workflow model.

This page compares executorlib with the tools scientists most often weigh it against, and is honest about when each
alternative is the better choice.

## At a glance

| | executorlib | Concurrent futures | Dask | Parsl | Ray |
|---|---|---|---|---|---|
| Drop-in `Executor` API | ✅ | ✅ | ⚠️ | ⚠️  | ❌ |
| Per-call resource assignment | ✅ | ❌ | ⚠️ | ✅ | ✅ |
| Native HPC scheduler (SLURM/flux) | ✅ | ❌ | ⚠️ | ✅ | ⚠️ |
| MPI-parallel functions | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ |
| Caching of results | ✅ | ❌ | ⚠️ | ✅ | ❌ |
| Setup / learning overhead | Low | Very low | Medium | Medium | Medium |

✅ first-class · ⚠️ possible via an add-on or extra configuration · ❌ not supported.

## [Concurrent futures](https://docs.python.org/3/library/concurrent.futures.html)

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

## Choose executorlib when

- You already have Python functions and want to scale them across HPC nodes with minimal rewriting.
- You want to assign cores, threads, or GPUs **per function call**.
- You want native [SLURM](https://slurm.schedmd.com) / [flux](http://flux-framework.org) integration and optional MPI
  parallelism inside your functions.
- You want optional caching of intermediate results for rapid, iterative prototyping in notebooks.
