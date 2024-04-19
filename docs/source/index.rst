====================================================================
pympipool - up-scale python functions for high performance computing
====================================================================

:Author:  Jan Janssen
:Contact: janssen@lanl.gov

Challenges
----------
In high performance computing (HPC) the Python programming language is commonly used as high-level language to
orchestrate the coupling of scientific applications. Still the efficient usage of highly parallel HPC clusters remains
challenging, in primarily three aspects:

* **Communication**: Distributing python function calls over hundreds of compute node and gathering the results on a shared file system is technically possible, but highly inefficient. A socket-based communication approach is preferable.
* **Resource Management**: Assigning Python functions to GPUs or executing Python functions on multiple CPUs using the message passing interface (MPI) requires major modifications to the python workflow.
* **Integration**: Existing workflow libraries implement a secondary the job management on the Python level rather than leveraging the existing infrastructure provided by the job scheduler of the HPC.

pympipool is ...
^^^^^^^^^^^^^^^^
In a given HPC allocation the :code:`pympipool` library addresses these challenges by extending the Executor interface
of the standard Python library to support the resource assignment in the HPC context. Computing resources can either be
assigned on a per function call basis or as a block allocation on a per Executor basis. The :code:`pympipool` library
is built on top of the `flux-framework <https://flux-framework.org>`_ to enable fine-grained resource assignment. In
addition `Simple Linux Utility for Resource Management (SLURM) <https://slurm.schedmd.com>`_ is supported as alternative
queuing system and for workstation installations :code:`pympipool` can be installed without a job scheduler.

pympipool is not ...
^^^^^^^^^^^^^^^^^^^^
The pympipool library is not designed to request an allocation from the job scheduler of an HPC. Instead within a given
allocation from the job scheduler the :code:`pympipool` library can be employed to distribute a series of python
function calls over the available computing resources to achieve maximum computing resource utilization.

Examples
--------
The following examples illustrates how :code:`pympipool` can be used to distribute an MPI parallel function within a
queuing system allocation using the `flux-framework <https://flux-framework.org>`_. :code:`example.py`::

    import flux.job
    from pympipool import Executor

    def calc(i):
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        return i, size, rank

    with flux.job.FluxExecutor() as flux_exe:
        with Executor(max_cores=2, cores_per_worker=2, executor=flux_exe) as exe:
            fs = exe.submit(calc, 3)
            print(fs.result())

This example can be executed using::

    python example.py

Which returns::

    [(0, 2, 0), (0, 2, 1)], [(1, 2, 0), (1, 2, 1)]

The important part in this example is that `mpi4py <https://mpi4py.readthedocs.io>`_ is only used in the :code:`calc()`
function, not in the python script, consequently it is not necessary to call the script with :code:`mpiexec` but instead
a call with the regular python interpreter is sufficient. This highlights how :code:`pympipool` allows the users to
parallelize one function at a time and not having to convert their whole workflow to use `mpi4py <https://mpi4py.readthedocs.io>`_.
At the same time the function can be distributed over all compute nodes in a given allocation of the job scheduler, in
contrast to the standard `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_
which only supports distribution within one compute node.

The interface of the standard `concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_
is extended by adding the option :code:`cores_per_worker=2` to assign multiple MPI ranks to each function call. To
create two workers the maximum number of cores can be increased to :code:`max_cores=4`. In this case each worker
receives two cores resulting in a total of four CPU cores being utilized.

After submitting the function :code:`calc()` with the corresponding parameter to the executor :code:`exe.submit(calc, 0)`
a python `concurrent.futures.Future <https://docs.python.org/3/library/concurrent.futures.html#future-objects>`_ is
returned. Consequently, the :code:`pympipool.Executor` can be used as a drop-in replacement for the
`concurrent.futures.Executor <https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures>`_
which allows the user to add parallelism to their workflow one function at a time.

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
