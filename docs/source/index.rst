====================================================================
pympipool - up-scale python functions for high performance computing
====================================================================

:Author:  Jan Janssen
:Contact: janssen@lanl.gov

Up-scaling python functions for high performance computing (HPC) can be challenging. While the python standard library
provides interfaces for multiprocessing and asynchronous task execution, namely
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ and
`concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_ both are
limited to the execution on a single compute node. So a series of python libraries have been developed to address the
up-scaling of python functions for HPC. Starting in the datascience and machine learning community with solutions
like `dask <https://www.dask.org>`_ over more HPC focused solutions like
`fireworks <https://materialsproject.github.io/fireworks/>`_ and `parsl <http://parsl-project.org>`_ up to Python
bindings for the message passing interface (MPI) named `mpi4py <https://mpi4py.readthedocs.io>`_. Each of these
solutions has their advantages and disadvantages, in particular scaling beyond serial python functions, including thread
based parallelism, MPI parallel python application or assignment of GPUs to individual python function remains
challenging.

To address these challenges :code:`pympipool` is developed with three goals in mind:

* Extend the standard python library `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_ interface, to minimize the barrier of up-scaling an existing workflow to be used on HPC resources.
* Integrate thread based parallelism, MPI parallel python functions based on `mpi4py <https://mpi4py.readthedocs.io>`_ and GPU assignment. This allows the users to accelerate their workflows one function at a time.
* Embrace `Jupyter <https://jupyter.org>`_ notebooks for the interactive development of HPC workflows, as they allow the users to document their though process right next to the python code and their results all within one document.

HPC Context
-----------
In contrast to frameworks like `dask <https://www.dask.org>`_, `fireworks <https://materialsproject.github.io/fireworks/>`_
and `parsl <http://parsl-project.org>`_ which can be used to submit a number of worker processes directly the the HPC
queuing system and then transfer tasks from either the login node or an interactive allocation to these worker processes
to accelerate the execution, `mpi4py <https://mpi4py.readthedocs.io>`_ and :code:`pympipool` follow a different
approach. Here the user creates their HPC allocation first and then `mpi4py <https://mpi4py.readthedocs.io>`_ or
:code:`pympipool` can be used to distribute the tasks within this allocation. The advantage of this approach is that
no central data storage is required as the workers and the scheduling task can communicate directly.

Examples
--------
The following examples illustrates how :code:`pympipool` can be used to distribute a series of MPI parallel function
calls within a queuing system allocation. :code:`example.py`::

    from pympipool import Executor

    def calc(i):
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        return i, size, rank

    with Executor(max_workers=2, cores_per_worker=2) as exe:
        fs_0 = exe.submit(calc, 0)
        fs_1 = exe.submit(calc, 1)
        print(fs_0.result(), fs_1.result())

This example can be executed using::

    python example.py

Which returns::

    [(0, 2, 0), (0, 2, 1)], [(1, 2, 0), (1, 2, 1)]

The important part in this example is that `mpi4py <https://mpi4py.readthedocs.io>`_ is only used in the :code:`calc()`
function, not in the python script, consequently it is not necessary to call the script with :code:`mpiexec` but instead
a call with the regular python interpreter is sufficient. This highlights how :code:`pympipool` allows the users to
parallelize one function at a time and not having to convert their whole workflow to use `mpi4py <https://mpi4py.readthedocs.io>`_.
The same code can also be executed inside a jupyter notebook directly which enables an interactive development process.

The standard `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_
interface is extended by adding the option :code:`cores_per_worker=2` to assign multiple MPI ranks to each function call.
To create two workers :code:`max_workers=2` each with two cores each requires a total of four CPU cores to be available.
After submitting the function :code:`calc()` with the corresponding parameter to the executor :code:`exe.submit(calc, 0)`
a python `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ is
returned. Consequently, the :code:`pympipool.Executor` can be used as a drop-in replacement for the
`concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_
which allows the user to add parallelism to their workflow one function at a time.

Backends
--------
Depending on the availability of different resource schedulers in your HPC environment the :code:`pympipool.Executor`
uses a different backend, with the :code:`pympipool.flux.PyFluxExecutor` being the preferred backend:

* :code:`pympipool.mpi.PyMpiExecutor`: The simplest executor of the three uses `mpi4py <https://mpi4py.readthedocs.io>`_ as a backend. This simplifies the installation on all operation systems including Windows. Still at the same time it limits the up-scaling to a single compute node and serial or MPI parallel python functions. There is no support for thread based parallelism or GPU assignment. This interface is primarily used for testing and developing or as a fall-back solution. It is not recommended to use this interface in production.
* :code:`pympipool.slurm.PySlurmExecutor`: The `SLURM workload manager <https://www.schedmd.com>`_ is commonly used on HPC systems to schedule and distribute tasks. :code:`pympipool` provides a python interface for scheduling the execution of python functions as SLURM job steps which are typically created using the :code:`srun` command. This executor supports serial python functions, thread based parallelism, MPI based parallelism and the assignment of GPUs to individual python functions. When the `SLURM workload manager <https://www.schedmd.com>`_ is installed on your HPC cluster this interface can be a reasonable choice, still depending on the SLURM configuration in can be limited in terms of the fine-grained scheduling or the responsiveness when working with hundreds of compute nodes in an individual allocation.
* :code:`pympipool.flux.PyFluxExecutor`: The `flux <https://flux-framework.org>`_ is the preferred backend for :code:`pympipool`. Just like the :code:`pympipool.slurm.PySlurmExecutor` it supports serial python functions, thread based parallelism, MPI based parallelism and the assignment of GPUs to individual python functions. Still the advantages of using the `flux <https://flux-framework.org>`_ as a backend are the easy installation, the faster allocation of resources as the resources are managed within the allocation and no central databases is used and the superior level of fine-grained resource assignment which is typically not available on HPC resource schedulers.

Each of these backends consists of two parts a broker and a worker. When a new tasks is submitted from the user it is
received by the broker and the broker identifies the first available worker. The worker then executes a task and returns
it to the broker, who returns it to the user. While there is only one broker per :code:`pympipool.Executor` the number
of workers can be specified with the :code:`max_workers` parameter.

Disclaimer
----------
While we try to develop a stable and reliable software library, the development remains a opensource project under the
BSD 3-Clause License without any warranties::

    BSD 3-Clause License

    Copyright (c) 2022, Jan Janssen
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation
   examples
   development

* :ref:`modindex`