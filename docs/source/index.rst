.. pympipool documentation master file, created by
   sphinx-quickstart on Sat Jun 10 11:15:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pympipool - scale python functions over multiple compute nodes
==============================================================

Up-scaling python functions for high performance computing (HPC) can be challenging. While the python standard library provides interfaces for multiprocessing and asynchronous task execution, namely `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ and `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_ both are limited to the execution on a single compute node. So a series of python libraries have been developed to address the up-scaling of python functions for HPC. Starting in the datascience and machine learning community with solutions like `dask <https://www.dask.org>`_ over more HPC focused solutions like `parsl <http://parsl-project.org>`_ up to Python bindings for the message passing interface (MPI) named `mpi4py <https://mpi4py.readthedocs.io>`_. Each of these solutions has their advantages and disadvantages, in particular the mixing of MPI parallel python functions and serial python functions in combined workflows remains challenging.

To address this challenges :code:`pympipool` is developed with three goals in mind:

* Reimplement the standard python library interfaces namely `multiprocessing.pool.Pool <https://docs.python.org/3/library/multiprocessing.html>`_ and `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_ as closely as possible, to minimize the barrier of up-scaling an existing workflow to be used on HPC resources.
* Integrate MPI parallel python functions based on `mpi4py <https://mpi4py.readthedocs.io>`_ on the same level as serial python functions, so both can be combined in a single workflow. This allows the users to parallelize their workflows one function at a time. Internally this is achieved by coupling a serial python process to a MPI parallel python process.
* Embrace `Jupyter <https://jupyter.org>`_ notebooks for the interactive development of HPC workflows, as they allow the users to document their though process right next to the python code and their results all within one document.

Features
--------
As different users and different workflows have different requirements in terms of the level of parallelization, the
:code:`pympipool` implements a series of five different interfaces:

* :code:`pympipool.Pool`: Following the `multiprocessing.pool.Pool <https://docs.python.org/3/library/multiprocessing.html>`_ the :code:`pympipool.Pool` class implements the `map()` and `starmap()` functions. Internally these connect to an MPI parallel subprocess running the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_. So by increasing the number of workers, by setting the :code:`max_workers` parameter the :code:`pympipool.Pool` can scale the execution of serial python functions beyond a single compute node. For MPI parallel python functions the :code:`pympipool.MPISpawnPool` is derived from the :code:`pympipool.Pool` and uses :code:`MPI_Spawn()` to execute those.
* :code:`pympipool.Executor`: The easiest way to execute MPI parallel python functions right next to serial python functions is the :code:`pympipool.Executor`. It implements the executor interface defined by the `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_. So functions are submitted to the :code:`pympipool.Executor` using the :code:`submit()` function, which returns an `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ object. With these `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ objects asynchronous workflows can constructed which periodically check if the computation is completed `done()` and then query the results using the :code:`result()` function. The limitation of the :code:`pympipool.Executor` is lack of load balancing, each :code:`pympipool.Executor` acts as a serial first in first out (FIFO) queue. So it is the task of the user to balance the load of many different tasks over multiple :code:`pympipool.Executor` instances.
* :code:`pympipool.PoolExecutor`: To combine the functionality of the :code:`pympipool.Pool` and the :code:`pympipool.Executor` the :code:`pympipool.PoolExecutor` again connects to the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_. Still in contrast to the :code:`pympipool.Pool` it does not implement the :code:`map()` and :code:`starmap()` functions but rather the :code:`submit()` function based on the `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_ interface. In this case the load balancing happens internally and the maximum number of workers :code:`max_workers` defines the maximum number of parallel tasks. But only serial python tasks can be executed in contrast to the :code:`pympipool.Executor` which can also execute MPI parallel python tasks.
* :code:`pympipool.MPISpawnPool`: An alternative way to support MPI parallel functions in addition to the :code:`pympipool.Executor` is the :code:`pympipool.MPISpawnPool`. Just like the :code:`pympipool.Pool` it supports the :code:`map()` and :code:`starmap()` functions. The additional :code:`ranks_per_task` parameter defines how many MPI ranks are used per task. All functions are executed with the same number of MPI ranks. The limitation of this approach is that it uses :code:`MPI_Spawn()` to create new MPI ranks for the execution of the individual tasks. Consequently, this approach is not as scalable as the :code:`pympipool.Executor` but it offers load balancing for a large number of similar MPI parallel tasks.
* :code:`pympipool.SocketInterface`: The key functionality of the :code:`pympipool` package is the coupling of a serial python process with an MPI parallel python process. This happens in the background using a combination of the `zero message queue <https://zeromq.org>`_ and `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ to communicate binary python objects. The :code:`pympipool.SocketInterface` is an abstraction of this interface, which is used in the other classes inside :code:`pympipool` and might also be helpful for other projects.

In addition to using MPI to start a number of processes on different HPC computing resources, :code:`pympipool` also supports the `flux-framework <https://flux-framework.org>`_ as additional backend. By setting the optional :code:`enable_flux_backend` parameter to :code:`True` the flux backend can be enabled for the :code:`pympipool.Pool`, :code:`pympipool.Executor` and :code:`pympipool.PoolExecutor`. Other optional parameters include the selection of the working directory where the python function should be executed :code:`cwd` and the option to oversubscribe MPI tasks which is an `OpenMPI <https://www.open-mpi.org>`_ specific feature which can be enabled by setting :code:`oversubscribe` to :code:`True`. For more details on the :code:`pympipool` classes and their application, the  extended documentation is linked below.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   interfaces
   comparison
   development
