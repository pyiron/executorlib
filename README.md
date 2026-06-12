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

## Examples
The Python standard library provides the [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)
with the [ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and the 
[ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor) for parallel 
execution of Python functions on a single computer. executorlib extends this functionality to distribute Python 
functions over multiple computers within a high performance computing (HPC) cluster. This can be either achieved by 
submitting each function as individual job to the HPC job scheduler with an [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) - 
or by requesting a job from the HPC cluster and then distribute the Python functions within this job with an
[HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html). Finally, to accelerate the 
development process executorlib also provides a [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html) - 
to use the executorlib functionality on a laptop, workstation or single compute node for testing. Starting with the 
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

## Documentation
* [Installation](https://executorlib.readthedocs.io/en/latest/installation.html)
  * [Minimal](https://executorlib.readthedocs.io/en/latest/installation.html#minimal)
  * [MPI Support](https://executorlib.readthedocs.io/en/latest/installation.html#mpi-support)
  * [Caching](https://executorlib.readthedocs.io/en/latest/installation.html#caching)
  * [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-cluster-executor)
  * [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-job-executor)
  * [Visualisation](https://executorlib.readthedocs.io/en/latest/installation.html#visualisation)
  * [For Developers](https://executorlib.readthedocs.io/en/latest/installation.html#for-developers)
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
  * [Flux](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#flux)
* [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html)
  * [SLURM](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm)
  * [SLURM with Flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm-with-flux)
  * [Flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#flux)
* [Application](https://executorlib.readthedocs.io/en/latest/application.html)
  * [GPAW](https://executorlib.readthedocs.io/en/latest/4-1-gpaw.html)
  * [Quantum Espresso](https://executorlib.readthedocs.io/en/latest/4-2-quantum-espresso.html)
* [Coupling with other Libraries](https://executorlib.readthedocs.io/en/latest/coupling.html)
  * [emcee](https://executorlib.readthedocs.io/en/latest/coupling.html#emcee-markov-chain-monte-carlo)
  * [pipefunc](https://executorlib.readthedocs.io/en/latest/coupling.html#pipefunc-function-pipelines)
  * [omp4py](https://executorlib.readthedocs.io/en/latest/coupling.html#omp4py-openmp-for-python)
  * [pylammpsmpi](https://executorlib.readthedocs.io/en/latest/coupling.html#pylammpsmpi-mpi-parallel-lammps)
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
