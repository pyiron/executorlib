# executorlib
[![Pipeline](https://github.com/pyiron/executorlib/actions/workflows/pipeline.yml/badge.svg)](https://github.com/pyiron/executorlib/actions/workflows/pipeline.yml)
[![codecov](https://codecov.io/gh/pyiron/executorlib/graph/badge.svg?token=KFIO3R08H3)](https://codecov.io/gh/pyiron/executorlib)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/executorlib/HEAD)
[![status](https://joss.theoj.org/papers/7a4bced702e5ea10e4630bc4d5dbaad2/status.svg)](https://joss.theoj.org/papers/7a4bced702e5ea10e4630bc4d5dbaad2)
![GitHub Repo stars](https://img.shields.io/github/stars/pyiron/executorlib)

Up-scale python functions for high performance computing (HPC) with executorlib. 

## Key Features
* **Up-scale your Python functions beyond a single computer.** - executorlib extends the [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)
  from the Python standard library and combines it with job schedulers for high performance computing (HPC) including 
  the [Simple Linux Utility for Resource Management (SLURM)](https://slurm.schedmd.com) and [flux](http://flux-framework.org). 
  With this combination executorlib allows users to distribute their Python functions over multiple compute nodes.
* **Parallelize your Python program one function at a time** - executorlib allows users to assign dedicated computing
  resources like CPU cores, threads or GPUs to one Python function call at a time. So you can accelerate your Python 
  code function by function.
* **Permanent caching of intermediate results to accelerate rapid prototyping** - To accelerate the development of 
  machine learning pipelines and simulation workflows executorlib provides optional caching of intermediate results for 
  iterative development in interactive environments like jupyter notebooks.

## Why executorlib?
executorlib is the lightest path to take *existing* Python functions and scale them across high performance computing
(HPC) nodes — with per-function-call resource control and native [SLURM](https://slurm.schedmd.com) and
[flux](http://flux-framework.org) integration — without rewriting your code into a new paradigm. It extends the standard
library [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) you already
know, rather than asking you to adopt a new data, actor, or workflow model.

| | executorlib | [Futures](https://docs.python.org/3/library/concurrent.futures.html) | [Dask](https://www.dask.org) | [Parsl](https://parsl-project.org) | [Ray](https://www.ray.io) |
|---|---|---|---|---|---|
| Drop-in `Executor` API | ✅ | ✅ | ⚠️ | ⚠️  | ❌ |
| Per-call resource assignment | ✅ | ❌ | ⚠️ | ✅ | ✅ |
| Native HPC scheduler (SLURM/flux) | ✅ | ❌ | ⚠️ | ✅ | ⚠️ |
| MPI-parallel functions | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ |
| Caching of results | ✅ | ❌ | ⚠️ | ✅ | ❌ |
| Setup / learning overhead | Low | Very low | Medium | Medium | Medium |

✅ first-class · ⚠️ possible via add-on/config · ❌ not supported. See the full
[comparison: when to use which](https://executorlib.readthedocs.io/en/latest/comparison.html) for honest guidance on
when another tool is the better fit.

## Getting Started on HPC
Running Python functions on an HPC cluster typically means writing batch scripts, requesting resources for each 
calculation, tracking job IDs to chain calculations, and resubmitting work after a session was interrupted. executorlib 
replaces these steps with the [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) 
of the Python standard library. Submitting a Python function from the login node as a SLURM job is as simple as:
```python
from executorlib import SlurmClusterExecutor


def calc(i):
    return i * 2


with SlurmClusterExecutor(cache_directory="./cache") as exe:
    future_lst = [exe.submit(calc, i) for i in range(1, 4)]
    print([f.result() for f in future_lst])
```
The user stories each start from a typical challenge of HPC users and solve it end-to-end, combining the features of
executorlib which are needed along the way:
1. [Submit a long-running function and come back later](https://executorlib.readthedocs.io/en/latest/story-1-submit-and-reconnect.html) - submit
   from the login node, disconnect, check the status, reconnect and cancel jobs.
2. [A pipeline whose steps need different resources](https://executorlib.readthedocs.io/en/latest/story-2-multi-resource-pipeline.html) - per-call
   resources, MPI-parallel functions and SLURM job dependencies from future objects.
3. [A parameter study with many short calculations](https://executorlib.readthedocs.io/en/latest/story-3-parameter-study.html) - distribute
   calculations within one allocation, find failing parameter sets and collect the results in a DataFrame.
4. [Prototype on a laptop, then scale up](https://executorlib.readthedocs.io/en/latest/story-4-prototype-then-scale.html) - develop locally with
   the same interface, test the file-based cluster mode and switch executors with one line.
5. [Recover from failed jobs](https://executorlib.readthedocs.io/en/latest/story-5-recover-failed-jobs.html) - diagnose exceptions and killed
   jobs, then rerun only the failed calculations.
6. [Pay for an expensive setup only once](https://executorlib.readthedocs.io/en/latest/story-6-expensive-setup-once.html) - long-lived workers
   with an initialization function and automatic restarts.
7. [Large jobs which each fan out into many small tasks](https://executorlib.readthedocs.io/en/latest/story-7-fan-out-inside-jobs.html) -
   combine one SLURM job per campaign with many tasks inside each job.
8. [Adaptive workflows](https://executorlib.readthedocs.io/en/latest/story-8-adaptive-workflows.html) - convergence loops, recursion with
   `asyncio` and batching in the order results arrive.
9. [Document and share a workflow](https://executorlib.readthedocs.io/en/latest/story-9-document-and-share.html) - export the workflow graph and
   use the cache as an audit trail.
10. [Parallel computations on pandas DataFrames](https://executorlib.readthedocs.io/en/latest/story-10-pandas-dataframes.html) -
    replace `df.apply()`, submit chunks of rows as SLURM jobs and recompute only new rows with the cache.
11. [Couple Python to simulation codes which use MPI internally](https://executorlib.readthedocs.io/en/latest/story-11-couple-mpi-simulation-codes.html) -
    wrap MPI executables written in C, C++ or Fortran in Python functions, each with its own working directory.
12. [Parallelize only the functions which need it with mpi4py](https://executorlib.readthedocs.io/en/latest/story-12-mpi4py-functions.html) -
    MPI-parallel functions within a serial script, their per-rank results and mpi4py-enabled codes like GPAW.

The cluster-specific settings like the partition, run time limits and the submission template are collected in one
cell at the top of each story. Readers without access to an HPC cluster can start with
[Story 4](https://executorlib.readthedocs.io/en/latest/story-4-prototype-then-scale.html) or follow the examples below locally.

## Examples
The Python standard library provides the [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)
with the [ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and the 
[ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor) for parallel 
execution of Python functions on a single computer. executorlib extends this functionality to distribute Python 
functions over multiple computers within a high performance computing (HPC) cluster. This can be either achieved by 
submitting each function as individual job to the HPC job scheduler with an [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) - 
or by requesting a job from the HPC cluster and then distribute the Python functions within this job with an
[HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html). To develop and test the same workflow 
on a laptop, workstation or single compute node before using cluster resources, executorlib also provides a 
[Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html). The following examples can be 
executed without cluster access, starting with the 
[Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html):
```python
from executorlib import SingleNodeExecutor


with SingleNodeExecutor() as exe:
    future_lst = [exe.submit(sum, [i, i]) for i in range(1, 5)]
    print([f.result() for f in future_lst])
```
In the same way executorlib can also execute Python functions which use additional computing resources, like multiple 
CPU cores, CPU threads or GPUs. For example if the Python function internally uses the Message Passing Interface (MPI) 
via the [mpi4py](https://mpi4py.readthedocs.io) Python libary: 
```python
from executorlib import SingleNodeExecutor


def calc(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


with SingleNodeExecutor() as exe:
    fs = exe.submit(calc, 3, resource_dict={"cores": 2})
    print(fs.result())
```
The additional `resource_dict` parameter defines the computing resources allocated to the execution of the submitted 
Python function. In addition to the compute cores `cores`, the resource dictionary can also define the threads per core
as `threads_per_core`, the GPUs per core as `gpus_per_core`, the working directory with `cwd`, the option to use the
OpenMPI oversubscribe feature with `openmpi_oversubscribe` and finally for the [Simple Linux Utility for Resource 
Management (SLURM)](https://slurm.schedmd.com) queuing system the option to provide additional command line arguments 
with the `slurm_cmd_args` parameter - [resource dictionary](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#resource-dictionary)
This flexibility to assign computing resources on a per-function-call basis simplifies the up-scaling of Python programs.
Only the part of the Python functions which benefit from parallel execution are implemented as MPI parallel Python 
funtions, while the rest of the program remains serial. 

The same function can be submitted to the [SLURM](https://slurm.schedmd.com) job scheduler by replacing the 
`SingleNodeExecutor` with the `SlurmClusterExecutor`.  The rest of the example remains the same, which highlights how 
executorlib accelerates the rapid prototyping and up-scaling of HPC Python programs. 
```python
from executorlib import SlurmClusterExecutor


def calc(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


with SlurmClusterExecutor() as exe:
    fs = exe.submit(calc, 3, resource_dict={"cores": 2})
    print(fs.result())
```
In this case the [Python simple queuing system adapter (pysqa)](https://pysqa.readthedocs.io) is used to submit the 
`calc()` function to the [SLURM](https://slurm.schedmd.com) job scheduler and request an allocation with two CPU cores 
for the execution of the function - [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html). In the background the [sbatch](https://slurm.schedmd.com/sbatch.html) 
command is used to request the allocation to execute the Python function. 

Within a given [SLURM](https://slurm.schedmd.com) job executorlib can also be used to assign a subset of the 
available computing resources to execute a given Python function. In terms of the [SLURM](https://slurm.schedmd.com) 
commands, this functionality internally uses the [srun](https://slurm.schedmd.com/srun.html) command to receive a subset
of the resources of a given queuing system allocation. 
```python
from executorlib import SlurmJobExecutor


def calc(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


with SlurmJobExecutor() as exe:
    fs = exe.submit(calc, 3, resource_dict={"cores": 2})
    print(fs.result())
```
In addition, to support for [SLURM](https://slurm.schedmd.com) executorlib also provides support for the hierarchical 
[flux](http://flux-framework.org) job scheduler. The [flux](http://flux-framework.org) job scheduler is developed at 
[Larwence Livermore National Laboratory](https://computing.llnl.gov/projects/flux-building-framework-resource-management)
to address the needs for the up-coming generation of Exascale computers. Still even on traditional HPC clusters the 
hierarchical approach of the [flux](http://flux-framework.org) is beneficial to distribute hundreds of tasks within a
given allocation. Even when [SLURM](https://slurm.schedmd.com) is used as primary job scheduler of your HPC, it is 
recommended to use [SLURM with flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm-with-flux) 
as hierarchical job scheduler within the allocations. 

## Which Executor should I use?
executorlib provides five `Executor` classes. They all share the same `submit()` / `map()` interface and only differ in
*where* the Python functions are executed and *how* the resources are requested. A common workflow is to develop and 
test with the `SingleNodeExecutor` on a laptop and then switch to one of the HPC executors by changing only the class 
name:

| Executor | Where it runs | Scheduler command | Best for |
|---|---|---|---|
| [`SingleNodeExecutor`](https://executorlib.readthedocs.io/en/latest/1-single-node.html) | laptop, workstation or single compute node | `subprocess` | developing and testing a workflow |
| [`SlurmClusterExecutor`](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#slurm) | HPC login node | `sbatch` (one job per function) | long-running functions that should outlive the Python session |
| [`SlurmJobExecutor`](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm) | inside a SLURM allocation | `srun` (job steps) | many functions within one existing allocation |
| [`FluxClusterExecutor`](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#flux) | HPC login node or Flux instance | `flux submit` | long-running functions; disconnecting and reconnecting |
| [`FluxJobExecutor`](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#flux) | inside a Flux allocation | `flux run` | high-throughput execution of many short functions |

The **Cluster** executors submit each Python function as an individual job and communicate via the file system, so the
Python process which created the executor can be closed and the results reloaded later. The **Job** executors run inside
an existing allocation and communicate via sockets, which has lower overhead and is the better choice for many short
function calls.

To set up the submission of Python functions on a new SLURM cluster, the [executorlib-cluster-setup](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#setup-with-claude-code)
skill for [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) analyses the cluster and generates a tested,
cluster-specific configuration for the `SlurmClusterExecutor`.

## Documentation
* [Installation](https://executorlib.readthedocs.io/en/latest/installation.html)
  * [Minimal](https://executorlib.readthedocs.io/en/latest/installation.html#minimal)
  * [MPI Support](https://executorlib.readthedocs.io/en/latest/installation.html#mpi-support)
  * [Caching](https://executorlib.readthedocs.io/en/latest/installation.html#caching)
  * [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-cluster-executor)
  * [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-job-executor)
  * [Visualisation](https://executorlib.readthedocs.io/en/latest/installation.html#visualisation)
  * [For Developers](https://executorlib.readthedocs.io/en/latest/installation.html#for-developers)
* User Stories
  * [Submit a Long-running Function and Come Back Later](https://executorlib.readthedocs.io/en/latest/story-1-submit-and-reconnect.html)
  * [A Pipeline whose Steps Need Different Resources](https://executorlib.readthedocs.io/en/latest/story-2-multi-resource-pipeline.html)
  * [A Parameter Study with Many Short Calculations](https://executorlib.readthedocs.io/en/latest/story-3-parameter-study.html)
  * [Prototype on a Laptop, then Scale Up](https://executorlib.readthedocs.io/en/latest/story-4-prototype-then-scale.html)
  * [Recover from Failed Jobs](https://executorlib.readthedocs.io/en/latest/story-5-recover-failed-jobs.html)
  * [Pay for an Expensive Setup Only Once](https://executorlib.readthedocs.io/en/latest/story-6-expensive-setup-once.html)
  * [Large Jobs which Each Fan Out into Many Small Tasks](https://executorlib.readthedocs.io/en/latest/story-7-fan-out-inside-jobs.html)
  * [Adaptive Workflows](https://executorlib.readthedocs.io/en/latest/story-8-adaptive-workflows.html)
  * [Document and Share a Workflow](https://executorlib.readthedocs.io/en/latest/story-9-document-and-share.html)
  * [Parallel Computations on pandas DataFrames](https://executorlib.readthedocs.io/en/latest/story-10-pandas-dataframes.html)
  * [Couple Python to Simulation Codes which Use MPI Internally](https://executorlib.readthedocs.io/en/latest/story-11-couple-mpi-simulation-codes.html)
  * [Parallelize Only the Functions which Need It with mpi4py](https://executorlib.readthedocs.io/en/latest/story-12-mpi4py-functions.html)
* [Comparison](https://executorlib.readthedocs.io/en/latest/comparison.html#comparison)
  * [At a glace](https://executorlib.readthedocs.io/en/latest/comparison.html#at-a-glance)
  * [Concurrent futures](https://executorlib.readthedocs.io/en/latest/comparison.html#concurrent-futures-the-python-standard-library)
  * [Dask](https://executorlib.readthedocs.io/en/latest/comparison.html#dask)
  * [Parsl](https://executorlib.readthedocs.io/en/latest/comparison.html#parsl)
  * [Ray](https://executorlib.readthedocs.io/en/latest/comparison.html#ray)
  * [Choose executorlib when](https://executorlib.readthedocs.io/en/latest/comparison.html#choose-executorlib-when)
* [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html)
  * [Basic Functionality](https://executorlib.readthedocs.io/en/latest/1-single-node.html#basic-functionality)
  * [Parallel Functions](https://executorlib.readthedocs.io/en/latest/1-single-node.html#parallel-functions)
  * [Performance Optimization](https://executorlib.readthedocs.io/en/latest/1-single-node.html#performance-optimization)
  * [Advanced Scheduling](https://executorlib.readthedocs.io/en/latest/1-single-node.html#advanced-scheduling)
  * [Testing and Debugging](https://executorlib.readthedocs.io/en/latest/1-single-node.html#testing-and-debugging)
* [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html)
  * [SLURM](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#slurm)
  * [Setup with Claude Code](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#setup-with-claude-code)
  * [Flux](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#flux)
  * [Disconnecting and Reconnecting](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#disconnecting-and-reconnecting)
  * [Combine both](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#combine-both)
  * [Cleaning Cache](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#cleaning-cache)
* [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html)
  * [SLURM](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm)
  * [SLURM with Flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm-with-flux)
  * [Flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#flux)
* [Application](https://executorlib.readthedocs.io/en/latest/application.html)
  * [GPAW](https://executorlib.readthedocs.io/en/latest/4-1-gpaw.html)
  * [Quantum Espresso](https://executorlib.readthedocs.io/en/latest/4-2-quantum-espresso.html)
* [Integration](https://executorlib.readthedocs.io/en/latest/coupling.html)
  * [emcee](https://executorlib.readthedocs.io/en/latest/coupling.html#emcee)
  * [pipefunc](https://executorlib.readthedocs.io/en/latest/coupling.html#pipefunc)
  * [omp4py](https://executorlib.readthedocs.io/en/latest/coupling.html#omp4py)
  * [pylammpsmpi](https://executorlib.readthedocs.io/en/latest/coupling.html#pylammpsmpi)
  * [General Pattern](https://executorlib.readthedocs.io/en/latest/coupling.html#general-pattern)
* [Trouble Shooting](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html)
  * [Filesystem Usage](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#filesystem-usage)
  * [Firewall Issues](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#firewall-issues)
  * [Message Passing Interface](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#message-passing-interface)
  * [Test Coverage for Integration Tests](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#test-coverage-for-integration-tests)
  * [Python Version](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#python-version)
  * [Resource Dictionary](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#resource-dictionary)
  * [SSH Connection](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#ssh-connection)
* [Support & Contribution](https://executorlib.readthedocs.io/en/latest/5-developer.html)
  * [Issues](https://executorlib.readthedocs.io/en/latest/5-developer.html#issues)
  * [Pull Requests](https://executorlib.readthedocs.io/en/latest/5-developer.html#pull-requests)
  * [License](https://executorlib.readthedocs.io/en/latest/5-developer.html#license)
  * [Modules](https://executorlib.readthedocs.io/en/latest/5-developer.html#modules)
  * [Interface Class Hierarchy](https://executorlib.readthedocs.io/en/latest/5-developer.html#interface-class-hierarchy)
  * [Execution Flow](https://executorlib.readthedocs.io/en/latest/5-developer.html#execution-flow)
  * [Test Environment](https://executorlib.readthedocs.io/en/latest/5-developer.html#test-environment)
  * [Communication](https://executorlib.readthedocs.io/en/latest/5-developer.html#communication)
  * [External Libraries](https://executorlib.readthedocs.io/en/latest/5-developer.html#external-libraries)
  * [External Executables](https://executorlib.readthedocs.io/en/latest/5-developer.html#external-executables)
* [Interface](https://executorlib.readthedocs.io/en/latest/api.html)
