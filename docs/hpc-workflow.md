# Getting Started on HPC

executorlib is designed for one job: running *existing* Python functions on a high performance computing (HPC) cluster
without rewriting them into scheduler scripts. This page walks through a typical HPC workflow and introduces each
feature of executorlib at the point where it solves a concrete problem:

1. [Submission](#submission) - run a Python function as a SLURM job instead of writing a batch script.
2. [Resources and MPI](#resources-and-mpi) - give each function call its own MPI ranks, CPU threads and GPUs.
3. [Dependencies](#dependencies) - chain calculations without coordinating job IDs and intermediate files by hand.
4. [Caching](#caching) - reuse completed calculations when you revise, interrupt or resume a workflow.
5. [Execution strategy](#execution-strategy) - decide between one scheduler job per function and many functions
   inside one allocation.
6. [Local development](#local-development) - develop and test the same workflow on a laptop before using cluster
   resources.

The examples are taken from the detailed chapters on the [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html), the
[HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html) and the [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html), which provide additional
background for each step.

```{admonition} Cluster-specific prerequisites
:class: important
The HPC examples on this page cannot be executed on readthedocs or on a laptop. They assume:

* access to the login node of an HPC cluster with the [SLURM](https://slurm.schedmd.com) job scheduler,
* executorlib installed with the [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-cluster-executor) dependencies
  ([pysqa](https://pysqa.readthedocs.io) and [h5py](https://www.h5py.org)) - plus [mpi4py](https://mpi4py.readthedocs.io)
  for the MPI examples - in a Python environment which is also available on the compute nodes,
* a shared file system between the login node and the compute nodes, which is used for the `cache_directory`.

Values marked as **cluster-specific** in the examples below - like the partition name `"regular"`, the run time limit or
the PMI mode `"pmix"` - have to be adapted to your cluster. The
[executorlib-cluster-setup](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#setup-with-claude-code) skill for
[Claude Code](https://docs.claude.com/en/docs/claude-code/overview) can analyse your cluster and generate a tested
configuration. Readers without cluster access can follow the same workflow with the `SingleNodeExecutor` as explained in
the [local development](#local-development) section.
```

## Submission
**Challenge:** You have a Python function which runs too long for the login node. The classical approach is to write a
Python script, wrap it in a SLURM batch script, submit it with `sbatch`, wait, and then parse the output files.

**Solution:** The `SlurmClusterExecutor` follows the [Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)
of the Python standard library. Each call of `submit()` writes the function and its arguments to the `cache_directory`,
submits one SLURM job via `sbatch` and returns a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
object. Calling `result()` on the future waits for the job to finish and loads the return value:

```python
from executorlib import SlurmClusterExecutor


def calc(i):
    return i * 2


with SlurmClusterExecutor(cache_directory="./cache") as exe:
    future_lst = [exe.submit(calc, i) for i in range(1, 4)]
    print([f.result() for f in future_lst])
```

The SLURM settings which are specific to your cluster are defined in a [Jinja2](https://jinja.palletsprojects.com)
submission template and passed via the resource dictionary `resource_dict`:

```python
submission_template = """\
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env=L
#SBATCH --partition={{partition}}
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if dependency %}
#SBATCH --dependency=afterok:{{ dependency | join(',') }}
{%- endif %}
#SBATCH --ntasks={{cores}}

{{command}}
"""

with SlurmClusterExecutor(
    cache_directory="./cache",
    resource_dict={
        "submission_template": submission_template,
        "partition": "regular",  # cluster-specific
        "run_time_max": 3600,  # in seconds, cluster-specific
    },
) as exe:
    future = exe.submit(calc, 4)
    print(future.result())
```

Rather than repeating the template in every script it can be stored once in a pysqa configuration directory and
referenced with the `pysqa_config_directory` parameter, as explained in
[reusing the configuration](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#reusing-the-configuration-with-pysqa-config-directory). When a
submission fails, for example because the default template does not match the cluster, `sbatch` returns an error which
is raised as `CalledProcessError` - see [SLURM](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#slurm) for details.

## Resources and MPI
**Challenge:** Most steps of a scientific workflow are serial, but a few are expensive simulations which should use many
CPU cores via MPI, multiple threads or GPUs. Writing the whole program in MPI is not an option, and requesting the
largest allocation for every step wastes resources.

**Solution:** executorlib assigns resources per function call. The resource dictionary `resource_dict` defines the
number of MPI ranks `cores`, the CPU threads per rank `threads_per_core` and the GPUs per rank `gpus_per_core`. Only the
function which benefits from parallel execution uses [mpi4py](https://mpi4py.readthedocs.io), the rest of the
program remains serial:

```python
from executorlib import SlurmClusterExecutor


def calc_mpi(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


with SlurmClusterExecutor(
    cache_directory="./cache",
    pmi_mode="pmix",  # cluster-specific, depends on the MPI library
    resource_dict={"submission_template": submission_template, "partition": "regular"},
) as exe:
    fs_serial = exe.submit(calc_mpi, 1, resource_dict={"cores": 1})
    fs_mpi = exe.submit(calc_mpi, 2, resource_dict={"cores": 4})
    fs_gpu = exe.submit(calc_mpi, 3, resource_dict={"cores": 1, "gpus_per_core": 1})
    print(fs_serial.result(), fs_mpi.result(), fs_gpu.result())
```

The `resource_dict` passed to `submit()` extends the default `resource_dict` defined for the executor. The full list of
supported keys is available in the [resource dictionary](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#resource-dictionary) section. To confirm
that SLURM assigned the requested resources, the job ID of each function is stored in the cache and can be inspected
with `sacct` - see [verifying the resource assignment](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#verifying-the-resource-assignment).

## Dependencies
**Challenge:** The output of one calculation is the input of the next one. With plain SLURM this requires collecting
job IDs, adding `--dependency=afterok:<jobid>` to each batch script and writing and parsing intermediate files.

**Solution:** Pass the future object of one calculation as an argument to the next `submit()` call. executorlib
translates the dependency into a SLURM job dependency and transfers the result between the jobs via the
`cache_directory`. The Python process does not have to wait for the first calculation to finish before submitting the
second one:

```python
from executorlib import SlurmClusterExecutor


def add_funct(a, b):
    return a + b


with SlurmClusterExecutor(cache_directory="./cache") as exe:
    future = 0
    for i in range(4, 8):
        future = exe.submit(add_funct, i, future)
    print(future.result())
```

Functions returning tuples or dictionaries can be split into individual futures with `split_future()` and
`get_item_from_future()`, and many futures can be grouped with `batched()`. These patterns are introduced in the
[advanced scheduling](https://executorlib.readthedocs.io/en/latest/1-single-node.html#advanced-scheduling) section and apply unchanged to the HPC executors. To
debug complex dependencies the graph can be visualised with `plot_dependency_graph=True`, see
[dependencies](https://executorlib.readthedocs.io/en/latest/1-single-node.html#dependencies).

## Caching
**Challenge:** HPC workflows are developed iteratively. You add an analysis step to a workflow whose simulations took
hours, your session on the login node is disconnected, or one step fails and has to be fixed. In each case you do not
want to repeat the calculations which already finished.

**Solution:** All results of the `SlurmClusterExecutor` are stored in the `cache_directory`. Each task is identified by
a hash of the function and its arguments. When the same function is submitted again with the same arguments and the
same `cache_directory`, the stored result is returned instead of submitting a new job. This also allows you to submit a
workflow, close the Python process and reconnect later:

```python
from executorlib import SlurmClusterExecutor

exe = SlurmClusterExecutor(cache_directory="./cache")
future_lst = [exe.submit(sum, [i, i]) for i in range(1, 4)]
exe.shutdown(wait=False, cancel_futures=False)  # disconnect, the jobs keep running
```

Later - even from a new Python process - the same submission reloads the results from the cache:

```python
with SlurmClusterExecutor(cache_directory="./cache") as exe:
    future_lst = [exe.submit(sum, [i, i]) for i in range(1, 4)]
    print([f.result() for f in future_lst])
```

The results of all cached calculations can be collected in a `pandas.DataFrame` with `get_cache_data()`, see
[cache](https://executorlib.readthedocs.io/en/latest/1-single-node.html#cache). Keep in mind the lifetime and limitations of the cache:

* **What invalidates a cached result:** changing the function or its arguments creates a new hash and a new
  calculation. Changing only the `resource_dict` does not, so a result computed with 4 cores is reused when the same
  call is submitted with 8 cores. An explicit identifier can be set with the `cache_key` entry of the
  [resource dictionary](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html#resource-dictionary).
* **Duplicates within one session:** the cache restores results from previous sessions. It is not designed to detect
  the same function call submitted twice within one `with`-statement.
* **Lifetime:** executorlib never deletes the `cache_directory`. Remove it once the results are no longer needed, as
  explained in [cleaning cache](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#cleaning-cache), since disk space on shared file systems is limited.
* **Not long-term storage:** the cache uses [cloudpickle](https://github.com/cloudpipe/cloudpickle) and
  [h5py](https://www.h5py.org). The format may change between versions of executorlib and Python, so results which
  should be kept have to be stored in a format of your choice.

## Execution strategy
**Challenge:** Submitting every function as its own SLURM job is robust, but each job waits in the queue and the
scheduler is not designed for thousands of short jobs. On the other hand many clusters favour large allocations, or you
already have an interactive allocation.

**Solution:** Use the `SlurmJobExecutor` inside an existing SLURM allocation. It distributes the functions over the
resources of the allocation using `srun` job steps, so no new jobs enter the queue. The `submit()` interface, the
`resource_dict` and the dependency handling remain the same:

```python
from executorlib import SlurmJobExecutor

# This code runs inside an existing SLURM allocation, e.g. a script submitted with sbatch or salloc
with SlurmJobExecutor(max_workers=4) as exe:
    future_lst = [exe.submit(calc_mpi, i, resource_dict={"cores": 2}) for i in range(4)]
    print([f.result() for f in future_lst])
```

Both strategies can also be combined: a function submitted with the `SlurmClusterExecutor` can itself create a
`SlurmJobExecutor` to distribute many small tasks within its allocation - see [combine both](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#combine-both).
For many short tasks within one allocation, starting the hierarchical [flux](http://flux-framework.org) scheduler inside
the SLURM allocation reduces the load on the SLURM database - see [SLURM with flux](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html#slurm-with-flux).

| | `SlurmClusterExecutor` | `SlurmJobExecutor` |
|---|---|---|
| Where the Python process runs | login node | inside a SLURM allocation |
| One function call is | one `sbatch` job | one `srun` job step |
| Waiting time | each function waits in the queue | the allocation waits once |
| Python process can be closed while functions run | yes, results are reloaded from the cache | no |
| Communication | shared file system (`cache_directory` required) | sockets (`cache_directory` optional) |
| Overhead per function | high (queue and file system) | low |
| Best for | few long-running functions with different resource requirements | many short functions, high throughput |

As a rule of thumb: start with the `SlurmClusterExecutor` when individual calculations run for minutes to hours or the
workflow should survive the end of your login session. Switch to the `SlurmJobExecutor` (or `FluxJobExecutor`) when you
execute many short functions or your cluster limits the number of jobs per user. The
[which executor should I use](https://executorlib.readthedocs.io/en/latest/#which-executor-should-i-use) table
summarizes all executors, including the corresponding [flux](http://flux-framework.org) executors.

## Local development
**Challenge:** Debugging a workflow by waiting in the queue of the cluster is slow and consumes compute time, and not
everybody has access to an HPC cluster.

**Solution:** Develop and test the workflow with the `SingleNodeExecutor` on your laptop or workstation. It supports
the same `submit()` interface, the same `resource_dict` - including MPI parallel functions with `cores` - the same
dependency handling and the same `cache_directory`. Moving to the cluster only requires replacing the executor class
and adding the cluster-specific settings:

```python
from executorlib import SingleNodeExecutor


def calc_mpi(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def add_funct(a, b):
    return a + b


with SingleNodeExecutor(cache_directory="./cache") as exe:
    fs_mpi = exe.submit(calc_mpi, 3, resource_dict={"cores": 2})
    fs_sum = exe.submit(add_funct, 1, 2)
    fs_final = exe.submit(add_funct, fs_sum, 3)  # dependency via the future object
    print(fs_mpi.result(), fs_final.result())
```

The `SingleNodeExecutor` communicates via sockets like the `SlurmJobExecutor`. To test the file based communication of
the `SlurmClusterExecutor` locally, executorlib provides the `TestClusterExecutor` - see
[TestClusterExecutor](https://executorlib.readthedocs.io/en/latest/1-single-node.html#testclusterexecutor). The full functionality of the local executor is
introduced in the [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html) chapter, which can be executed without cluster access, for
example on [mybinder](https://mybinder.org/v2/gh/pyiron/executorlib/HEAD).

## Next steps
* [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) - SLURM background, submission templates, pysqa configuration, flux and
  disconnecting and reconnecting.
* [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html) - executing functions inside an allocation with SLURM and flux.
* [Application](https://executorlib.readthedocs.io/en/latest/application.html) - complete examples for [GPAW](https://executorlib.readthedocs.io/en/latest/4-1-gpaw.html) and
  [Quantum Espresso](https://executorlib.readthedocs.io/en/latest/4-2-quantum-espresso.html).
* [Trouble Shooting](https://executorlib.readthedocs.io/en/latest/trouble_shooting.html) - file system usage, firewalls, MPI and the full resource dictionary.
