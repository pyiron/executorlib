# pympipool
[![Unittests](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml/badge.svg)](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/pympipool/badge.svg?branch=main)](https://coveralls.io/github/pyiron/pympipool?branch=main)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pyiron/pympipool/HEAD?labpath=notebooks%2Fexamples.ipynb)

## Challenges
In high performance computing (HPC) the Python programming language is commonly used as high-level language to
orchestrate the coupling of scientific applications. Still the efficient usage of highly parallel HPC clusters remains
challenging, in primarily three aspects:

* **Communication**: Distributing python function calls over hundreds of compute node and gathering the results on a
  shared file system is technically possible, but highly inefficient. A socket-based communication approach is 
  preferable.
* **Resource Management**: Assigning Python functions to GPUs or executing Python functions on multiple CPUs using the 
  message passing interface (MPI) requires major modifications to the python workflow.
* **Integration**: Existing workflow libraries implement a secondary the job management on the Python level rather than
  leveraging the existing infrastructure provided by the job scheduler of the HPC.

### pympipool is ...
In a given HPC allocation the `pympipool` library addresses these challenges by extending the Executor interface
of the standard Python library to support the resource assignment in the HPC context. Computing resources can either be
assigned on a per function call basis or as a block allocation on a per Executor basis. The `pympipool` library
is built on top of the [flux-framework](https://flux-framework.org) to enable fine-grained resource assignment. In
addition, [Simple Linux Utility for Resource Management (SLURM)](https://slurm.schedmd.com) is supported as alternative
queuing system and for workstation installations `pympipool` can be installed without a job scheduler.

### pympipool is not ...
The pympipool library is not designed to request an allocation from the job scheduler of an HPC. Instead within a given
allocation from the job scheduler the `pympipool` library can be employed to distribute a series of python
function calls over the available computing resources to achieve maximum computing resource utilization.

## Example
The following examples illustrates how `pympipool` can be used to distribute a series of MPI parallel function calls 
within a queuing system allocation. `example.py`:
```python
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
```
This example can be executed using:
```
python example.py
```
Which returns:
```
>>> [(0, 2, 0), (0, 2, 1)], [(1, 2, 0), (1, 2, 1)]
```
The important part in this example is that [mpi4py](https://mpi4py.readthedocs.io) is only used in the `calc()`
function, not in the python script, consequently it is not necessary to call the script with `mpiexec` but instead
a call with the regular python interpreter is sufficient. This highlights how `pympipool` allows the users to
parallelize one function at a time and not having to convert their whole workflow to use [mpi4py](https://mpi4py.readthedocs.io).
The same code can also be executed inside a jupyter notebook directly which enables an interactive development process.

The interface of the standard [concurrent.futures.Executor](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
is extended by adding the option `cores_per_worker=2` to assign multiple MPI ranks to each function call. To create two 
workers the maximum number of cores can be increased to `max_cores=4`. In this case each worker receives two cores
resulting in a total of four CPU cores being utilized.

After submitting the function `calc()` with the corresponding parameter to the executor `exe.submit(calc, 0)`
a python [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) is
returned. Consequently, the `pympipool.Executor` can be used as a drop-in replacement for the
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
which allows the user to add parallelism to their workflow one function at a time.

## Disclaimer
While we try to develop a stable and reliable software library, the development remains a opensource project under the
BSD 3-Clause License without any warranties::
```
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
```

# Documentation
* [Installation](https://pympipool.readthedocs.io/en/latest/installation.html)
  * [Compatible Job Schedulers](https://pympipool.readthedocs.io/en/latest/installation.html#compatible-job-schedulers)
  * [pympipool with Flux Framework](https://pympipool.readthedocs.io/en/latest/installation.html#pympipool-with-flux-framework)
  * [Test Flux Framework](https://pympipool.readthedocs.io/en/latest/installation.html#test-flux-framework)
  * [Without Flux Framework](https://pympipool.readthedocs.io/en/latest/installation.html#without-flux-framework)
* [Examples](https://pympipool.readthedocs.io/en/latest/examples.html)
  * [Compatibility](https://pympipool.readthedocs.io/en/latest/examples.html#compatibility)
  * [Resource Assignment](https://pympipool.readthedocs.io/en/latest/examples.html#resource-assignment)
  * [Data Handling](https://pympipool.readthedocs.io/en/latest/examples.html#data-handling)
  * [Up-Scaling](https://pympipool.readthedocs.io/en/latest/examples.html#up-scaling)
  * [Coupled Functions](https://pympipool.readthedocs.io/en/latest/examples.html#coupled-functions)
  * [SLURM Job Scheduler](https://pympipool.readthedocs.io/en/latest/examples.html#slurm-job-scheduler) 
  * [Workstation Support](https://pympipool.readthedocs.io/en/latest/examples.html#workstation-support)
* [Development](https://pympipool.readthedocs.io/en/latest/development.html)
  * [Contributions](https://pympipool.readthedocs.io/en/latest/development.html#contributions)
  * [License](https://pympipool.readthedocs.io/en/latest/development.html#license)
  * [Integration](https://pympipool.readthedocs.io/en/latest/development.html#integration)
