# Coupling with other Libraries
A lot of scientific Python packages already know how to distribute work over many processes - they just need to be
handed an object that behaves like an executor or a worker pool. Because executorlib implements the
[Executor interface](https://docs.python.org/3/library/concurrent.futures.html#executor-objects) of the Python standard
library, it can be passed to these packages in place of the
[ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) or the
[multiprocessing.Pool](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool). The practical
benefit for you as a scientist is that the workflow stays exactly the same when you move from your laptop to a high
performance computer (HPC): you develop and test with the [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html)
and then switch to the [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) or the
[HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html) by changing a single class name.

```{note}
The examples below require the respective third-party package to be installed in addition to executorlib (for example
`pip install emcee`). They are shown as reference code rather than executed examples, because these optional packages are
not part of the executorlib test environment. In every example the `SingleNodeExecutor` can be replaced by a
`FluxJobExecutor`, `FluxClusterExecutor`, `SlurmJobExecutor` or `SlurmClusterExecutor` to scale the same workflow to an
HPC cluster.
```

## emcee (Markov Chain Monte Carlo)
[emcee](https://emcee.readthedocs.io) is a widely used Python package for Markov Chain Monte Carlo (MCMC) sampling, for
example to estimate the posterior distribution of model parameters from experimental data. The likelihood function has
to be evaluated many times per sampling step, and these evaluations are independent of each other, so they can be
executed in parallel. The `EnsembleSampler` of emcee accepts any worker pool which provides a `map()` function via the
`pool` parameter. As executorlib provides this interface, an executorlib `Executor` can be used directly as a drop-in
replacement for the [multiprocessing.Pool](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool)
which is recommended in the [emcee parallelization tutorial](https://emcee.readthedocs.io/en/stable/tutorials/parallel/):
```python
import numpy as np
import emcee
from executorlib import SingleNodeExecutor


def log_prob(theta):
    return -0.5 * np.sum(theta**2)


initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape

with SingleNodeExecutor(block_allocation=True) as exe:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=exe)
    sampler.run_mcmc(initial, 100, progress=True)
```
Here the `block_allocation=True` parameter is set to reuse the same Python processes for the repeated evaluation of the
`log_prob()` function, which reduces the overhead for these many short function calls -
[block allocation](https://executorlib.readthedocs.io/en/latest/1-single-node.html#block-allocation). For computationally
more expensive likelihood functions the parallel evaluation provides a substantial speed-up, and by replacing the
`SingleNodeExecutor` with a `FluxJobExecutor` the very same sampling can be distributed over multiple compute nodes of an
HPC cluster.

## pipefunc (Function Pipelines)
[pipefunc](https://pipefunc.readthedocs.io) is a library to build function pipelines, where the output of one function
is used as the input for the next function, including map-reduce patterns over many parameters. pipefunc takes care of
the book keeping of the pipeline, while the actual execution of the individual functions is delegated to an executor.
The `map()` function of a pipefunc `Pipeline` accepts an `executor` parameter, which can either be a single executorlib
`Executor` or a dictionary which assigns a dedicated executor to each output:
```python
import numpy as np
from pipefunc import Pipeline, pipefunc
from executorlib import SingleNodeExecutor


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def f(x):
    return x**2


@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def g(y):
    return y + 1


pipeline = Pipeline([f, g])
inputs = {"x": [1, 2, 3]}

with SingleNodeExecutor() as exe:
    results = pipeline.map(inputs, executor=exe)
    print(results["z"].output.tolist())
```
Assigning a different executor to each output enables fine-grained control over the computing resources. For example a
serial preprocessing step can be executed on a single core while a computationally expensive simulation step is
distributed over multiple compute nodes:
```python
executor = {
    "y": SingleNodeExecutor(max_workers=2),
    "z": SingleNodeExecutor(max_workers=4),
}
results = pipeline.map(inputs, executor=executor)
```
The combination of pipefunc and executorlib is explained in more detail in the
[pipefunc documentation on execution and parallelism](https://pipefunc.readthedocs.io/en/latest/concepts/execution-and-parallelism/).

## omp4py (OpenMP for Python)
The [thread based parallelism](https://executorlib.readthedocs.io/en/latest/1-single-node.html#thread-parallel-functions)
of executorlib is most commonly used to control the number of threads in linked libraries like NumPy. With
[omp4py](https://omp4py.readthedocs.io) - a Python implementation of [OpenMP](https://www.openmp.org) - it is also
possible to write thread parallel Python code directly. The number of threads assigned to the Python function is set via
the `threads_per_core` parameter in the `resource_dict`. The following example approximates the value of pi using a
parallel for loop with an OpenMP reduction:
```python
import random
from omp4py import omp
from executorlib import SingleNodeExecutor


@omp
def calc_pi(num_points):
    count = 0
    with omp("parallel for reduction(+:count)"):
        for i in range(num_points):
            x = random.random()
            y = random.random()
            if x * x + y * y <= 1.0:
                count += 1
    return 4 * (count / num_points)


with SingleNodeExecutor() as exe:
    future = exe.submit(calc_pi, 10000000, resource_dict={"threads_per_core": 4})
    print(future.result())
```
The `threads_per_core` parameter sets the environment variables which control the number of threads, so the requested
number of cores is reserved for the threads created by omp4py inside the `calc_pi()` function.

## pylammpsmpi (MPI-parallel LAMMPS)
[pylammpsmpi](https://pylammpsmpi.readthedocs.io) provides a Python interface to the molecular dynamics code
[LAMMPS](https://www.lammps.org) which distributes the simulation over multiple MPI ranks while the Python process
itself remains serial. Internally pylammpsmpi uses an executor to start the MPI-parallel LAMMPS processes, so an
executorlib `Executor` can be provided via the `executor` parameter. In combination with
[atomistics](https://atomistics.readthedocs.io) this can be used to run an MPI-parallel molecular dynamics simulation:
```python
from ase.build import bulk
from atomistics.calculators import (
    calc_molecular_dynamics_nvt_with_lammpslib,
    get_potential_by_name,
)
from pylammpsmpi import LammpsASELibrary
from executorlib import SingleNodeExecutor

structure = bulk("Ti")
potential = get_potential_by_name(potential_name="2016--Mendelev-M-I--Ti-3--LAMMPS--ipr1")

with SingleNodeExecutor(resource_dict={"cores": 2}) as exe:
    lmp = LammpsASELibrary(executor=exe)
    result_dict = calc_molecular_dynamics_nvt_with_lammpslib(
        structure=structure,
        potential_dataframe=potential,
        lmp=lmp,
    )
    lmp.close()
```
The `resource_dict={"cores": 2}` assigns two MPI ranks to the LAMMPS simulation. As for the other examples, replacing
the `SingleNodeExecutor` with one of the HPC executors distributes the LAMMPS simulation over the compute nodes of an
HPC cluster without any further changes to the simulation code.

## General Pattern
The four examples above follow the same pattern: a library which already supports parallel execution accepts an
executorlib `Executor` (or worker pool), so executorlib takes over the distribution of the work. Whenever a Python
package accepts a [concurrent.futures.Executor](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)
or a [multiprocessing.Pool](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool) - typically
exposed via a parameter named `executor` or `pool` - it can be combined with executorlib. The recommended approach
remains the same in all cases:

* Develop and test the workflow with the [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html)
  on a laptop or workstation.
* Switch to the [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html) or the
  [HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) to scale to an HPC cluster by
  changing only the executor class name.
