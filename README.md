# pympipool - scale python functions over multiple compute nodes
[![Unittests](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml/badge.svg)](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/pympipool/badge.svg?branch=main)](https://coveralls.io/github/pyiron/pympipool?branch=main)

Up-scaling python functions for high performance computing (HPC) can be challenging. While the python standard library 
provides interfaces for multiprocessing and asynchronous task execution, namely [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html)
and [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures) both are
limited to the execution on a single compute node. So a series of python libraries have been developed to address the 
up-scaling of python functions for HPC. Starting in the datascience and machine learning community with solutions like 
[dask](https://www.dask.org) over more HPC focused solutions like [parsl](http://parsl-project.org) up to Python bindings
for the message passing interface (MPI) named [mpi4py](https://mpi4py.readthedocs.io). Each of these solutions has their
advantages and disadvantages, in particular the mixing of MPI parallel python functions and serial python functions in
combined workflows remains challenging. 

To address these challenges `pympipool` is developed with three goals in mind: 
* Reimplement the standard python library interfaces namely [`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html)
and [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures) 
as closely as possible, to minimize the barrier of up-scaling an existing workflow to be used on HPC resources. 
* Integrate MPI parallel python functions based on [mpi4py](https://mpi4py.readthedocs.io) on the same level as serial 
python functions, so both can be combined in a single workflow. This allows the users to parallelize their workflows 
one function at a time. Internally this is achieved by coupling a serial python process to a MPI parallel python process.
* Embrace [Jupyter](https://jupyter.org) notebooks for the interactive development of HPC workflows, as they allow the
users to document their though process right next to the python code and their results all within one document. 

# Features 
As different users and different workflows have different requirements in terms of the level of parallelization, the 
`pympipool` implements a series of five different interfaces: 
* [`pympipool.Pool`](https://pympipool.readthedocs.io/en/latest/interfaces.html#pool): Following the 
[`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html) the `pympipool.Pool` class 
implements the `map()` and `starmap()` functions. Internally these connect to an MPI parallel subprocess running the 
[`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
So by increasing the number of workers, by setting the `max_workers` parameter the `pympipool.Pool` can scale the 
execution of serial python functions beyond a single compute node. For MPI parallel python functions the `pympipool.MPISpawnPool`
is derived from the `pympipool.Pool` and uses `MPI_Spawn()` to execute those. For more details see below. 
* [`pympipool.Executor`](https://pympipool.readthedocs.io/en/latest/interfaces.html#executor): The easiest way to 
execute MPI parallel python functions right next to serial python functions is the `pympipool.Executor`. It implements 
the executor interface defined by the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures).
So functions are submitted to the `pympipool.Executor` using the `submit()` function, which returns an 
[`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object. With 
these [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) objects 
asynchronous workflows can be constructed which periodically check if the computation is completed `done()` and then
query the results using the `result()` function. The limitation of the `pympipool.Executor` is lack of load balancing, 
each `pympipool.Executor` acts as a serial first in first out (FIFO) queue. So it is the task of the user to balance the
load of many different tasks over multiple `pympipool.Executor` instances. 
* [`pympipool.PoolExecutor`](https://pympipool.readthedocs.io/en/latest/interfaces.html#poolexecutor): To combine the 
functionality of the `pympipool.Pool` and the `pympipool.Executor` the `pympipool.PoolExecutor` again connects to the
[`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
Still in contrast to the `pympipool.Pool` it does not implement the `map()` and `starmap()` functions but rather the 
`submit()` function based on the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
interface. In this case the load balancing happens internally and the maximum number of workers `max_workers` defines
the maximum number of parallel tasks. But only serial python tasks can be executed in contrast to the `pympipool.Executor`
which can also execute MPI parallel python tasks. 
* [`pympipool.MPISpawnPool`](https://pympipool.readthedocs.io/en/latest/interfaces.html#mpispawnpool): An alternative 
way to support MPI parallel functions in addition to the `pympipool.Executor` is the `pympipool.MPISpawnPool`. Just like
the `pympipool.Pool` it supports the `map()` and `starmap()` functions. The additional `ranks_per_task` parameter 
defines how many MPI ranks are used per task. All functions are executed with the same number of MPI ranks. The 
limitation of this approach is that it uses `MPI_Spawn()` to create new MPI ranks for the execution of the individual 
tasks. Consequently, this approach is not as scalable as the `pympipool.Executor` but it offers load balancing for a
large number of similar MPI parallel tasks. 
* [`pympipool.SocketInterface`](https://pympipool.readthedocs.io/en/latest/interfaces.html#socketinterface): The key 
functionality of the `pympipool` package is the coupling of a serial python process with an MPI parallel python process.
This happens in the background using a combination of the [zero message queue](https://zeromq.org) and 
[cloudpickle](https://github.com/cloudpipe/cloudpickle) to communicate binary python objects. The `pympipool.SocketInterface` 
is an abstraction of this interface, which is used in the other classes inside `pympipool` and might also be helpful for
other projects. 

In addition to using MPI to start a number of processes on different HPC computing resources, `pympipool` also supports
the [flux-framework](https://flux-framework.org) as additional backend. By setting the optional `enable_flux_backend` 
parameter to `True` the flux backend can be enabled for the `pympipool.Pool`, `pympipool.Executor` and `pympipool.PoolExecutor`.
Other optional parameters include the selection of the working directory where the python function should be executed `cwd`
and the option to oversubscribe MPI tasks which is an [OpenMPI](https://www.open-mpi.org) specific feature which can be 
enabled by setting `oversubscribe` to `True`. For more details on the `pympipool` classes and their application, the 
extended documentation is linked below. 

# Documentation
* [Installation](https://pympipool.readthedocs.io/en/latest/installation.html) 
  * [pypi-based installation](https://pympipool.readthedocs.io/en/latest/installation.html#pypi-based-installation)
  * [conda-based installation](https://pympipool.readthedocs.io/en/latest/installation.html#conda-based-installation)
* [Interfaces](https://pympipool.readthedocs.io/en/latest/interfaces.html) 
  * [Pool](https://pympipool.readthedocs.io/en/latest/interfaces.html#pool)
  * [Executor](https://pympipool.readthedocs.io/en/latest/interfaces.html#executor)
  * [ParallelExecutor](https://pympipool.readthedocs.io/en/latest/interfaces.html#poolexecutor)
  * [MPISpawnPool](https://pympipool.readthedocs.io/en/latest/interfaces.html#mpispawnpool)
  * [SocketInterface](https://pympipool.readthedocs.io/en/latest/interfaces.html#socketinterface)
* [Development](https://pympipool.readthedocs.io/en/latest/development.html) 

# License
`pympipool` is released under the BSD license https://github.com/pyiron/pympipool/blob/main/LICENSE . It is a spin-off of the `pyiron` project https://github.com/pyiron/pyiron therefore if you use `pympipool` for calculation which result in a scientific publication, please cite: 

    @article{pyiron-paper,
      title = {pyiron: An integrated development environment for computational materials science},
      journal = {Computational Materials Science},
      volume = {163},
      pages = {24 - 36},
      year = {2019},
      issn = {0927-0256},
      doi = {https://doi.org/10.1016/j.commatsci.2018.07.043},
      url = {http://www.sciencedirect.com/science/article/pii/S0927025618304786},
      author = {Jan Janssen and Sudarsan Surendralal and Yury Lysogorskiy and Mira Todorova and Tilmann Hickel and Ralf Drautz and Jörg Neugebauer},
      keywords = {Modelling workflow, Integrated development environment, Complex simulation protocols},
    }
