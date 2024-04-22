# Examples
The `pympipool.Executor` extends the interface of the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
to simplify the up-scaling of individual functions in a given workflow.

## Compatibility
Starting with the basic example of `1+1=2`. With the `ThreadPoolExecutor` from the [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
standard library this can be written as - `test_thread.py`: 
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=1) as exe:
    future = exe.submit(sum, [1, 1])
    print(future.result())
```
In this case `max_workers=1` limits the number of threads used by the `ThreadPoolExecutor` to one. Then the `sum()`
function is submitted to the executor with a list with two ones `[1, 1]` as input. A [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
object is returned. The `Future` object allows to check the status of the execution with the `done()` method which 
returns `True` or `False` depending on the state of the execution. Or the main process can wait until the execution is 
completed by calling `result()`. 

This example stored in a python file named `test_thread.py` can be executed using the python interpreter: 
```
python test_thread.py
>>> 2
```
The result of the calculation is `1+1=2`. 

The `pympipool.Executor` class extends the interface of the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures) 
class by providing more parameters to specify the level of parallelism. In addition, to specifying the maximum number 
of workers `max_workers` the user can also specify the number of cores per worker `cores_per_worker` for MPI based 
parallelism, the number of threads per core `threads_per_core` for thread based parallelism and the number of GPUs per
worker `gpus_per_worker`. Finally, for those backends which support over-subscribing this can also be enabled using the 
`oversubscribe` parameter. All these parameters are optional, so the `pympipool.Executor` can be used as a drop-in 
replacement for the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures).

The previous example is rewritten for the `pympipool.Executor` in - `test_sum.py`:
```python
import flux.job
from pympipool import Executor

with flux.job.FluxExecutor() as flux_exe:
    with Executor(max_cores=1, executor=flux_exe) as exe:
        future = exe.submit(sum, [1,1])
        print(future.result())
```
Again this example can be executed with the python interpreter: 
```
python test_sum.py
>>> 2
```
The result of the calculation is again `1+1=2`.

Beyond pre-defined functions like the `sum()` function, the same functionality can be used to submit user-defined 
functions. In the `test_serial.py` example a custom summation function is defined: 
```python
import flux.job
from pympipool import Executor

def calc(*args):
    return sum(*args)

with flux.job.FluxExecutor() as flux_exe:
    with Executor(max_cores=2, executor=flux_exe) as exe:
        fs_1 = exe.submit(calc, [2, 1])
        fs_2 = exe.submit(calc, [2, 2])
        fs_3 = exe.submit(calc, [2, 3])
        fs_4 = exe.submit(calc, [2, 4])
        print([
            fs_1.result(), 
            fs_2.result(), 
            fs_3.result(), 
            fs_4.result(),
        ])
```
In contrast to the previous example where just a single function was submitted to a single worker, in this case a total
of four functions is submitted to a group of two workers `max_workers=2`. Consequently, the functions are executed as a
set of two pairs. 

The script can be executed with any python interpreter:
```
python test_serial.py
>>> [3, 4, 5, 6]
```
It returns the corresponding sums as expected. The same can be achieved with the built-in [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
classes. Still one advantage of using the `pympipool.Executor` rather than the built-in ones, is the ability to execute 
the same commands in interactive environments like [Jupyter notebooks](https://jupyter.org). This is achieved by using 
[cloudpickle](https://github.com/cloudpipe/cloudpickle) to serialize the python function and its parameters rather than
the regular pickle package. 

For backwards compatibility with the [`multiprocessing.Pool`](https://docs.python.org/3/library/multiprocessing.html) 
class the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
also implements the `map()` function to map a series of inputs to a function. The same `map()` function is also 
available in the `pympipool.Executor` - `test_map.py`: 
```python
import flux.job
from pympipool import Executor

def calc(*args):
    return sum(*args)

with flux.job.FluxExecutor() as flux_exe:
    with Executor(max_cores=2, executor=flux_exe) as exe:
        print(list(exe.map(calc, [[2, 1], [2, 2], [2, 3], [2, 4]])))
```
Again the script can be executed with any python interpreter:
```
python test_map.py
>>> [3, 4, 5, 6]
```
The results remain the same. 

## Resource Assignment
By default, every submission of a python function results in a flux job (or SLURM job step) depending on the backend. 
This is sufficient for function calls which take several minutes or longer to execute. For python functions with shorter 
run-time `pympipool` provides block allocation (enabled by the `block_allocation=True` parameter) to execute multiple 
python functions with similar resource requirements in the same flux job (or SLURM job step). 

The following example illustrates the resource definition on both level. This is redundant. For block allocations the 
resources have to be configured on the **Executor level**, otherwise it can either be defined on the **Executor level**
or on the **Submission level**. The resource defined on the **Submission level** overwrite the resources defined on the 
**Executor level**.

```python
import flux.job
from pympipool import Executor


def calc_function(parameter_a, parameter_b):
    return parameter_a + parameter_b


with flux.job.FluxExecutor() as flux_exe:
    with Executor(  
        # Resource definition on the executor level
        max_cores=2,  # total number of cores available to the Executor
        # Optional resource definition 
        cores_per_worker=1,
        threads_per_core=1,
        gpus_per_worker=0,
        oversubscribe=False,  # not available with flux
        cwd="/home/jovyan/notebooks",
        executor=flux_exe,
        hostname_localhost=False,  # only required on MacOS
        backend="flux",  # optional in case the backend is not recognized
        block_allocation=False, 
        init_function=None,  # only available with block_allocation=True
        command_line_argument_lst=[],  # additional command line arguments for SLURM
        disable_dependencies=False,  # option to disable input dependency resolution
        refresh_rate=0.01,  # refresh frequency in seconds for the dependency resolution 
    ) as exe:
        future_obj = exe.submit(
            calc_function, 
            1,   # parameter_a
            parameter_b=2, 
            # Resource definition on the submission level
            resource_dict={
                "cores": 1,
                "threads_per_core": 1,
                "gpus_per_core": 0,  # here it is gpus_per_core rather than gpus_per_worker
                "oversubscribe": False,  # not available with flux
                "cwd": "/home/jovyan/notebooks",
                "executor": flux_exe,
                "hostname_localhost": False,  # only required on MacOS
                # "command_line_argument_lst": [],  # additional command line arguments for SLURM
            },
        )
        print(future_obj.result())
```
The `max_cores` which defines the total number of cores of the allocation, is the only mandatory parameter. All other
resource parameters are optional. If none of the submitted Python function uses [mpi4py](https://mpi4py.readthedocs.io)
or any GPU, then the resources can be defined on the **Executor level** as: `cores_per_worker=1`, `threads_per_core=1` 
and `gpus_per_worker=0`. These are defaults, so they do even have to be specified. In this case it also makes sense to 
enable `block_allocation=True` to continuously use a fixed number of python processes rather than creating a new python
process for each submission. In this case the above example can be reduced to: 
```python
import flux.job
from pympipool import Executor


def calc_function(parameter_a, parameter_b):
    return parameter_a + parameter_b


with flux.job.FluxExecutor() as flux_exe:
    with Executor(  
        # Resource definition on the executor level
        max_cores=2,  # total number of cores available to the Executor
        block_allocation=True,  # reuse python processes
        executor=flux_exe,
    ) as exe:
        future_obj = exe.submit(
            calc_function, 
            1,   # parameter_a
            parameter_b=2, 
        )
        print(future_obj.result())
```
The working directory parameter `cwd` can be helpful for tasks which interact with the file system to define which task
is executed in which folder, but for most python functions it is not required.

## Data Handling
A limitation of many parallel approaches is the overhead in communication when working with large datasets. Instead of
reading the same dataset repetitively, the `pympipool.Executor` in block allocation mode (`block_allocation=True`) 
loads the dataset only once per worker and afterward, each function submitted to this worker has access to the dataset, 
as it is already loaded in memory. To achieve this the user defines an initialization function `init_function` which 
returns a dictionary with one key per dataset. The keys of the dictionary can then be used as additional input 
parameters in each function submitted to the `pympipool.Executor`. When block allocation is disabled this functionality 
is not available, as each function is executed in a separate process, so no data can be preloaded.

This functionality is illustrated in the `test_data.py` example: 
```python
import flux.job
from pympipool import Executor

def calc(i, j, k):
    return i + j + k

def init_function():
    return {"j": 4, "k": 3, "l": 2}

with flux.job.FluxExecutor() as flux_exe:
    with Executor(max_cores=1, init_function=init_function, executor=flux_exe, block_allocation=True) as exe:
        fs = exe.submit(calc, 2, j=5)
        print(fs.result())
```
The function `calc()` requires three inputs `i`, `j` and `k`. But when the function is submitted to the executor only 
two inputs are provided `fs = exe.submit(calc, 2, j=5)`. In this case the first input parameter is mapped to `i=2`, the
second input parameter is specified explicitly `j=5` but the third input parameter `k` is not provided. So the 
`pympipool.Executor` automatically checks the keys set in the `init_function()` function. In this case the returned 
dictionary `{"j": 4, "k": 3, "l": 2}` defines `j=4`, `k=3` and `l=2`. For this specific call of the `calc()` function,
`i` and `j` are already provided so `j` is not required, but `k=3` is used from the `init_function()` and as the `calc()`
function does not define the `l` parameter this one is also ignored. 

Again the script can be executed with any python interpreter:
```
python test_data.py
>>> 10
```
The result is `2+5+3=10` as `i=2` and `j=5` are provided during the submission and `k=3` is defined in the `init_function()`
function.

## Up-Scaling 
[flux](https://flux-framework.org) provides fine-grained resource assigment via `libhwloc` and `pmi`.

### Thread-based Parallelism
The number of threads per core can be controlled with the `threads_per_core` parameter during the initialization of the 
`pympipool.Executor`. Unfortunately, there is no uniform way to control the number of cores a given underlying library 
uses for thread based parallelism, so it might be necessary to set certain environment variables manually: 

* `OMP_NUM_THREADS`: for openmp
* `OPENBLAS_NUM_THREADS`: for openblas
* `MKL_NUM_THREADS`: for mkl
* `VECLIB_MAXIMUM_THREADS`: for accelerate on Mac Os X
* `NUMEXPR_NUM_THREADS`: for numexpr

At the current stage `pympipool.Executor` does not set these parameters itself, so you have to add them in the function
you submit before importing the corresponding library: 

```python
def calc(i):
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    import numpy as np
    return i
```

Most modern CPUs use hyper-threading to present the operating system with double the number of virtual cores compared to
the number of physical cores available. So unless this functionality is disabled `threads_per_core=2` is a reasonable 
default. Just be careful if the number of threads is not specified it is possible that all workers try to access all 
cores at the same time which can lead to poor performance. So it is typically a good idea to monitor the CPU utilization
with increasing number of workers. 

Specific manycore CPU models like the Intel Xeon Phi processors provide a much higher hyper-threading ration and require
a higher number of threads per core for optimal performance. 

### MPI Parallel Python Functions
Beyond thread based parallelism, the message passing interface (MPI) is the de facto standard parallel execution in 
scientific computing and the [`mpi4py`](https://mpi4py.readthedocs.io) bindings to the MPI libraries are commonly used
to parallelize existing workflows. The limitation of this approach is that it requires the whole code to adopt the MPI
communication standards to coordinate the way how information is distributed. Just like the `pympipool.Executor` the 
[`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor) 
implements the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
interface. Still in this case eah python function submitted to the executor is still limited to serial execution. The
novel approach of the `pympipool.Executor` is mixing these two types of parallelism. Individual functions can use
the [`mpi4py`](https://mpi4py.readthedocs.io) library to handle the parallel execution within the context of this 
function while these functions can still me submitted to the `pympipool.Executor` just like any other function. The
advantage of this approach is that the users can parallelize their workflows one function at the time. 

The example in `test_mpi.py` illustrates the submission of a simple MPI parallel python function: 
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
The `calc()` function initializes the [`mpi4py`](https://mpi4py.readthedocs.io) library and gathers the size of the 
allocation and the rank of the current process within the MPI allocation. This function is then submitted to an 
`pympipool.Executor` which is initialized with a single worker with two cores `cores_per_worker=2`. So each function
call is going to have access to two cores. 

Just like before the script can be called with any python interpreter even though it is using the [`mpi4py`](https://mpi4py.readthedocs.io)
library in the background it is not necessary to execute the script with `mpiexec` or `mpirun`:
```
python test_mpi.py
>>> [(3, 2, 0), (3, 2, 1)]
```
The response consists of a list of two tuples, one for each MPI parallel process, with the first entry of the tuple 
being the parameter `i=3`, followed by the number of MPI parallel processes assigned to the function call `cores_per_worker=2`
and finally the index of the specific process `0` or `1`. 

### GPU Assignment
With the rise of machine learning applications, the use of GPUs for scientific application becomes more and more popular.
Consequently, it is essential to have full control over the assignment of GPUs to specific python functions. In the 
`test_gpu.py` example the `tensorflow` library is used to identify the GPUs and return their configuration: 
```python
import socket
import flux.job
from pympipool import Executor
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [
        (x.name, x.physical_device_desc, socket.gethostname()) 
        for x in local_device_protos if x.device_type == 'GPU'
    ]

with flux.job.FluxExecutor() as flux_exe:
    with Executor(
        max_workers=2, 
        gpus_per_worker=1,
        executor=flux_exe,
    ) as exe:
        fs_1 = exe.submit(get_available_gpus)
        fs_2 = exe.submit(get_available_gpus)
        print(fs_1.result(), fs_2.result())
```
The additional parameter `gpus_per_worker=1` specifies that one GPU is assigned to each worker. This functionality 
requires `pympipool` to be connected to a resource manager like the [SLURM workload manager](https://www.schedmd.com)
or preferably the [flux framework](https://flux-framework.org). The rest of the script follows the previous examples, 
as two functions are submitted and the results are printed. 

To clarify the execution of such an example on a high performance computing (HPC) cluster using the [SLURM workload manager](https://www.schedmd.com)
the submission script is given below: 
```shell
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --get-user-env=L

python test_gpu.py
```
The important part is that for using the `pympipool.slurm.PySlurmExecutor` backend the script `test_gpu.py` does not 
need to be executed with `srun` but rather it is sufficient to just execute it with the python interpreter. `pympipool`
internally calls `srun` to assign the individual resources to a given worker. 

For the more complex setup of running the [flux framework](https://flux-framework.org) as a secondary resource scheduler
within the [SLURM workload manager](https://www.schedmd.com) it is essential that the resources are passed from the 
[SLURM workload manager](https://www.schedmd.com) to the [flux framework](https://flux-framework.org). This is achieved
by calling `srun flux start` in the submission script: 
```shell
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --get-user-env=L

srun flux start python test_gpu.py
```
As a result the GPUs available on the two compute nodes are reported: 
```
>>> [('/device:GPU:0', 'device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0', 'cn138'),
>>>  ('/device:GPU:0', 'device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0', 'cn139')]
```
In this case each compute node `cn138` and `cn139` is equipped with one `Tesla V100S-PCIE-32GB`.

## Coupled Functions 
For submitting two functions with rather different computing resource requirements it is essential to represent this 
dependence during the submission process. In `pympipool` this can be achieved by leveraging the separate submission of 
individual python functions and including the `concurrent.futures.Future` object of the first submitted function as 
input for the second function during the submission. Consequently, this functionality can be used for directed acyclic 
graphs, still it does not enable cyclic graphs. As a simple example we can add one to the result of the addition of one
and two:
```python
from pympipool import Executor

def calc_function(parameter_a, parameter_b):
    return parameter_a + parameter_b

with flux.job.FluxExecutor() as flux_exe:
    with Executor(max_cores=2, executor=flux_exe) as exe:
        future_1 = exe.submit(
            calc_function, 
            parameter_a=1,
            parameter_b=2, 
        )
        future_2 = exe.submit(
            calc_function, 
            parameter_a=1,
            parameter_b=future_1, 
        )
        print(future_2.result())
```
Here the first addition `1+2` is computed and the output `3` is returned as the result of `future_1.result()`. Still 
before the computation of this addition is completed already the next addition is submitted which uses the future object
as an input `future_1` and adds `1`. The result of both additions is `4` as `1+2+1=4`. 

To disable this functionality the parameter `disable_dependencies=True` can be set on the executor level. Still at the
current stage the performance improvement of disabling this functionality seem to be minimal. Furthermore, this 
functionality introduces the `refresh_rate=0.01` parameter, it defines the refresh rate in seconds how frequently the 
queue of submitted functions is queried. Typically, there is no need to change these default parameters. 

## SLURM Job Scheduler
Using `pympipool` without the [flux framework](https://flux-framework.org) results in one `srun` call per worker in 
`block_allocation=True` mode and one `srun` call per submitted function in `block_allocation=False` mode. As each `srun`
call represents a request to the central database of SLURM this can drastically reduce the performance, especially for
large numbers of small python functions. That is why the hierarchical job scheduler [flux framework](https://flux-framework.org)
is recommended as secondary job scheduler even within the context of the SLURM job manager. 

Still the general usage of `pympipool` remains similar even with SLURM as backend:
```python
from pympipool import Executor

with Executor(max_cores=1, backend="slurm") as exe:
    future = exe.submit(sum, [1,1])
    print(future.result())
```
The `backend="slurm"` parameter is optional as `pympipool` automatically recognizes if [flux framework](https://flux-framework.org) 
or SLURM are available. 

In addition, the SLURM backend introduces the `command_line_argument_lst=[]` parameter, which allows the user to provide
a list of command line arguments for the `srun` command. 

## Workstation Support
While the high performance computing (HPC) setup is limited to the Linux operating system, `pympipool` can also be used
in combination with MacOS and Windows. These setups are limited to a single compute node. 

Still the general usage of `pympipool` remains similar:
```python
from pympipool import Executor

with Executor(max_cores=1, backend="mpi") as exe:
    future = exe.submit(sum, [1,1])
    print(future.result())
```
The `backend="mpi"` parameter is optional as `pympipool` automatically recognizes if [flux framework](https://flux-framework.org) 
or SLURM are available. 

Workstations, especially workstations with MacOs can have rather strict firewall settings. This includes limiting the
look up of hostnames and communicating with itself via their own hostname. To directly connect to `localhost` rather
than using the hostname which is the default for distributed systems, the `hostname_localhost=True` parameter is 
introduced. 
