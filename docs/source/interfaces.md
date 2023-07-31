# Interfaces
The `pympipool` class provides five different interfaces to scale python functions over multiple compute nodes. They are
briefly summarized here and explained in more detail below. 

|        Feature         | Pool | Executor | HPCExecutor | PoolExecutor | MPISpawnPool |
|:----------------------:|:----:|:--------:|:-----------:|:------------:|:------------:|
|        `map()`         | yes  |   yes    |     yes     |     yes      |     yes      |
|      `starmap()`       | yes  |    no    |     no      |      no      |     yes      |
|       `submit()`       |  no  |   yes    |     yes     |     yes      |      no      |
|   parallel execution   | yes  |    no    |     yes     |     yes      |     yes      |
| MPI parallel functions |  no  |   yes    |     yes     |      no      |     yes      |
| flux framework support | yes  |   yes    |     yes     |     yes      |      no      |
|    internal storage    |  no  |   yes    |     yes     |      no      |      no      | 

While all four interfaces implement the `map()` function, only two of them implement the `starmap()` function while the
rest implements the asynchronous `sumbit()` function which returns [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects).
In terms of the execution it is important to differentiate between parallel execution, meaning multiple individual 
functions are executed in parallel and MPI parallel functions, which each require multiple MPI ranks to be executed. 
Furthermore, most interfaces are integrated with the [flux-framework](https://flux-framework.org) and the 
[SLURM queuing system](https://slurm.schedmd.com) so rather than using MPI ranks to distribute functions over multiple
compute nodes, they can also use the flux-framework or the SLURM queuing system for this purpose. Finally, the 
`pympipool.Executor` and `pympipool.HPCExecutor` are currently the only interfaces which implements an internal storage,
so data can remain in the executor process while applying multiple functions which interact with this data. 

The sixth interface is the `SocketInterface`. This interface connects two python processes to transfer python objects 
between them. It is used for all the above interfaces to connect the serial python process of the user interacts, with
the MPI parallel python process, which executes the python functions over multiple compute nodes.  

## Pool
Following the [`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html) 
the `pympipool.Pool` class implements the `map()` and `starmap()` functions. Internally these connect to an MPI parallel
subprocess running the [`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
So by increasing the number of workers, by setting the `max_workers` parameter the `pympipool.Pool` can scale the 
execution of serial python functions beyond a single compute node. For MPI parallel python functions the `pympipool.MPISpawnPool`
is derived from the `pympipool.Pool` and uses `MPI_Spawn()` to execute those. For more details see below.

Example how to use the `pympipool.Pool` class. This can be executed inside a jupyter notebook, interactive python shell
or as a python script. For the example a python script is used. Write a python test script named `test_pool_map.py`: 
```python
import numpy as np
from pympipool import Pool

def calc(i):
    return np.array(i ** 2)

with Pool(max_workers=2) as p:
    print(p.map(func=calc, iterable=[1, 2, 3, 4]))
```
The function `calc()` is applied on the list of arguments `iterable`. The script is executed as serial python process,
while internally it uses MPI to execute two sets of two parameters at a time. As you see the `numpy` library is 
dynamically included when the function is transferred to the MPI parallel subprocess for execution. To execute the 
python file `test_pool.py` in a serial python process use: 
```
python test_pool_map.py
>>> [array(1), array(4), array(9), array(16)]
```
Beyond the number of workers defined by `max_workers`, the additional parameters are `oversubscribe` to enable 
[OpenMPI](https://www.open-mpi.org) over-subscription, `enable_flux_backend` and `enable_slurm_backend` to switch from 
MPI as backend to flux or SLURM as alternative backend. In addition, the parameters `queue_adapter` and 
`queue_adapter_kwargs` provide an interface to [pysqa](https://pysqa.readthedocs.org) the simple queue adapter for 
python. The `queue_adapter` can be set as `pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` 
parameter represents a dictionary of input arguments for the `submit_job()` function of the queue adapter. Finally, the
`cwd` parameter specifies the current working directory where the python functions are executed.

In addition to the `map()` function, the `pympipool.Pool` interface implements the `starmap()` function. The example is
very similar to the one above. Just this time the `calc()` function accepts two arguments rather than one: 
```python
from pympipool import Pool

def calc(i, j):
    return i + j

with Pool(max_workers=2) as p:
    print(p.starmap(func=calc, iterable=[[1, 2], [3, 4], [5, 6], [7, 8]]))
```
The script named `test_pool_starmap.py` is executed and the sum of the input parameters is returned: 
```
python test_pool_starmap.py
>>> [3, 7, 11, 15]
```
In summary the `pympipool.Pool` class implements both the `map()` function and the `starmap()` function to scale serial 
python functions over multiple compute nodes. It internally handles the load distribution over multiple compute nodes.

## Executor
The easiest way to execute MPI parallel python functions right next to serial python functions is the `pympipool.Executor`.
It implements the executor interface defined by the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures).
So functions are submitted to the `pympipool.Executor` using the `submit()` function, which returns an 
[`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) object. With 
these [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects) objects 
asynchronous workflows can be constructed which periodically check if the computation is completed `done()` and then query 
the results using the `result()` function. The limitation of the `pympipool.Executor` is lack of load balancing, each 
`pympipool.Executor` acts as a serial first in first out (FIFO) queue. So it is the task of the user to balance the load
of many different tasks over multiple `pympipool.Executor` instances. 

In comparison to the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
in the standard python library the `pympipool.Executor` can execute MPI parallel python functions which internally use
the `mpi4py` library. In this example the `calc()` function returns the total number of MPI ranks and the index of the
individual MPI ranks. By setting the `cores` parameter of the `pympipool.Executor` to `2` the `calc()` function is 
executed with two MPI ranks.
```python
from pympipool import Executor

def calc(i):
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank

with Executor(cores=2) as p:
    fs = p.submit(calc, 3)
    print(fs.result())
```
The important part is that in contrast to the `mpi4py` library the scripts which use the `pympipool.Executor` class can
be executed as serial python scripts, without the need to invoke external MPI calls. 
```
python test_executor_mpi.py
>>> [(3, 2, 0), (3, 2, 1)]
```
The responses of the individual MPI ranks are returned as a combined python list. So in this case each MPI rank returns
a triple of the parameter `i=3`, the total number of MPI ranks `2` and the index of the selected MPI rank. 

In addition to the ability to execute MPI parallel functions the `pympipool.Executor` class also implements an internal
data storage, which can be utilized for serial and MPI parallel python functions. By adding an initialization function
`init_function` as additional parameter to the initialization of the `pympipool.Executor` class which returns a dictionary
of python variables, these variables are added to the internal storage. Each function which is submitted to this 
`pympipool.Executor` class can use these variables as input parameters, interact with them or modify them. 
```python
from pympipool import Executor

def calc(i, j, k):
    return i + j + k

def init_function():
    return {"j": 4, "k": 3, "l": 2}

with Executor(cores=1, init_function=init_function) as p:
    fs = p.submit(calc, 2, j=5)
    print(fs.result())
```
In this example the `calc()` function takes three arguments `i`,`j` and `k`. While the arguments `j`, `k` and `l` are 
set by the `init_function()` function. When the `calc()` function is submitted only the `i` parameter is required, while
the parameters `j` and `k` can be accessed from the internal storage. At the same time the `l` parameter which is not 
used by the `calc()` function, does not interact with it. So not all functions have to use all parameters. Finally, 
when the parameter is provided during the submission `submit()`, like in this case the `j` parameter, then the submitted
parameter is used rather than the parameter from internal memory.
```
python test_executor_init.py
>>> 10
```
So the sum of `i`,`j` and `k` results in `10` rather than `9`. Beyond the number of cores defined by `cores`, the number
of GPUs defined by `gpus_per_task` and the initialization function defined by `init_function` the additional parameters
are `oversubscribe` to enable [OpenMPI](https://www.open-mpi.org) over-subscription, `enable_flux_backend` and 
`enable_slurm_backend` to switch from MPI as backend to flux or SLURM as alternative backend. In addition, the 
parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to [pysqa](https://pysqa.readthedocs.org) the
simple queue adapter for python. The `queue_adapter` can be set as `pysqa.queueadapter.QueueAdapter` object and the 
`queue_adapter_kwargs` parameter represents a dictionary of input arguments for the `submit_job()` function of the queue
adapter. Finally, the `cwd` parameter specifies the current working directory where the python functions are executed.

When multiple functions are submitted to the `pympipool.Executor` class then they are executed following the first in
first out principle. The `len()` function applied on the `pympipool.Executor` object can be used to list how many items
are still waiting to be executed. 

## HPCExecutor
To address the limitation of the `pympipool.Executor` that only a single task is executed at any time, the 
`pympipool.HPCExecutor` provides a wrapper around multiple `pympipool.Executor` objects. It balances the queues of the 
individual `pympipool.Executor` objects to maximize the throughput for the given resources. This functionality comes 
with an additional overhead of another thread, acting as a broker between the task queue of the `pympipool.HPCExecutor`
and the individual `pympipool.Executor` objects. 

Example how to use the `pympipool.HPCExecutor` class. This can be executed inside a jupyter notebook, interactive python 
shell or as a python script. For the example a python script is used. Write a python test script named `test_hpc_gpu.py`: 
```
import socket
from pympipool import HPCExecutor
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [
        (x.name, x.physical_device_desc, socket.gethostname()) 
        for x in local_device_protos if x.device_type == 'GPU'
    ]

with HPCExecutor(
    max_workers=2, 
    cores_per_worker=1, 
    gpus_per_worker=1, 
    enable_flux_backend=True,
) as exe:
    fs_1 = exe.submit(get_available_gpus)
    fs_2 = exe.submit(get_available_gpus)

print(fs_1.result())
print(fs_2.result())
```
The example demonstrates how one GPU is assigned to each of the two tasks which are executed in parallel. To access the
GPUs the `tensorflow` package is used in the `get_available_gpus()` function to return a brief summary of the available 
GPU. The initialization of the `pympipool.HPCExecutor` then follows the same scheme as the initialization of the 
`pympipool.Executor`. The `max_workers` argument defines the number of `pympipool.Executor` objects the 
`pympipool.HPCExecutor` is managing internally. Then for each of these `pympipool.Executor` objects the number of cores
is defined by `cores_per_worker` and the number of GPUs is defined by `gpus_per_worker`. By default the number of GPUs 
is set to zero, as assigning GPUs to tasks requires an advanced scheduling backend like the flux-framework enabled by 
the `enable_flux_backend` option or the SLURM queuing system backend enabled by the `enable_slurm_backend` option. In
addition, the parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to 
[pysqa](https://pysqa.readthedocs.org) the simple queue adapter for python. The `queue_adapter` can be set as 
`pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` parameter represents a dictionary of input 
arguments for the `submit_job()` function of the queue adapter. Finally, the `cwd` parameter specifies the current 
working directory where the python functions are executed.

The submission of the individual tasks follows the definition of the `pympipool.HPCExecutor` in analogy to the example
above for the `pympipool.Executor`. Finally, the results are printed to the standard output:
```
python test_hpc_gpu.py
>>> [('/device:GPU:0', 'device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0', 'cn138')]
>>> [('/device:GPU:0', 'device: 0, name: Tesla V100S-PCIE-32GB, pci bus id: 0000:84:00.0, compute capability: 7.0', 'cn139')]
```
The output highlights that for each of the two workers there is only a single GPU visible. This applies to the example
case where only one GPU is available on each of the two hosts, as well as to hosts with multiple GPUs. Consequently, 
the `pympipool.HPCExecutor` drastically simplifies the scheduling of `GPU` and their assignment for python tasks. 

## PoolExecutor
To combine the functionality of the `pympipool.Pool` and the `pympipool.Executor` the `pympipool.PoolExecutor` again
connects to the [`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
Still in contrast to the `pympipool.Pool` it does not implement the `map()` and `starmap()` functions but rather the 
`submit()` function based on the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
interface. In this case the load balancing happens internally and the maximum number of workers `max_workers` defines
the maximum number of parallel tasks. But only serial python tasks can be executed in contrast to the `pympipool.Executor`
which can also execute MPI parallel python tasks. 

In the example a simple `calc()` function which calculates the sum of two parameters `i` and `j` is submitted with four 
different parameter combinations to an `pympipool.PoolExecutor` with a total of two workers specified by the `max_workers`
parameter.
```python
from pympipool import PoolExecutor

def calc(i, j):
    return i + j

with PoolExecutor(max_workers=2) as p:
    fs1 = p.submit(calc, 1, 2)
    fs2 = p.submit(calc, 3, 4)
    fs3 = p.submit(calc, 5, 6)
    fs4 = p.submit(calc, 7, 8)
    print(fs1.result(), fs2.result(), fs3.result(), fs4.result())
```
The functions are executed in two sets of two function and the result is returned when all functions are executed as the
`result()` call waits until the future object completed.
```
python test_pool_executor.py
>>> 3 7 11 15
```
Beyond the number of workers defined by `max_workers`, the additional parameters are `oversubscribe` to enable 
[OpenMPI](https://www.open-mpi.org) over-subscription, `enable_flux_backend` and `enable_slurm_backend` to switch from 
MPI as backend to flux or SLURM as alternative backend. In addition, the parameters `queue_adapter` and 
`queue_adapter_kwargs` provide an interface to [pysqa](https://pysqa.readthedocs.org) the simple queue adapter for 
python. The `queue_adapter` can be set as `pysqa.queueadapter.QueueAdapter` object and the `queue_adapter_kwargs` 
parameter represents a dictionary of input arguments for the `submit_job()` function of the queue adapter. Finally, the
`cwd` parameter specifies the current working directory where the python functions are executed.

## MPISpawnPool
An alternative way to support MPI parallel functions in addition to the `pympipool.Executor` is the `pympipool.MPISpawnPool`. 
Just like the `pympipool.Pool` it supports the `map()` and `starmap()` functions. The additional `ranks_per_task` 
parameter defines how many MPI ranks are used per task. All functions are executed with the same number of MPI ranks. 
The limitation of this approach is that it uses `MPI_Spawn()` to create new MPI ranks for the execution of the 
individual tasks. Consequently, this approach is not as scalable as the `pympipool.Executor` but it offers load 
balancing for a large number of similar MPI parallel tasks. 

In the example the maximum number of workers is defined by the maximum number of MPI ranks `max_ranks` devided by the
number of ranks per task `ranks_per_tasks`. So in the case of a total of four ranks and two ranks per task only two 
workers are created. 
```python
from pympipool import MPISpawnPool

def calc(i, comm):
    return i, comm.Get_size(), comm.Get_rank()

with MPISpawnPool(max_ranks=4, ranks_per_task=2) as p:
    print(p.map(func=calc, iterable=[1, 2, 3, 4]))
```
In contrast to the `pympipool.Executor` which returns the results of each individual MPI rank, the `pympipool.MPISpawnPool`
only returns the results of one MPI rank per function call, so it is the users task to synchronize the response of the
MPI parallel functions. 
```
python test_mpispawnpool.py
>>> [[1, 2, 0], [2, 2, 0], [3, 2, 0], [4, 2, 0]]
```
Beyond the maximum number of ranks defined by `max_ranks` and the ranks per task defined by `ranks_per_task` the 
additional parameters are `oversubscribe` to enable [OpenMPI](https://www.open-mpi.org) over-subscription. In addition,
the parameters `queue_adapter` and `queue_adapter_kwargs` provide an interface to [pysqa](https://pysqa.readthedocs.org) 
the simple queue adapter for python. The `queue_adapter` can be set as `pysqa.queueadapter.QueueAdapter` object and the
`queue_adapter_kwargs` parameter represents a dictionary of input arguments for the `submit_job()` function of the queue
adapter. Finally, the `cwd` parameter specifies the current working directory  where the MPI parallel python functions 
are executed. The flux backend as well as the SLURM backend are not supported for the `pympipool.MPISpawnPool` as the 
`MPI_Spawn()` command is incompatible to the internal management of ranks inside flux and SLURM. 

## SocketInterface
`pympipool.SocketInterface`: The key functionality of the `pympipool` package is the coupling of a serial python process
with an MPI parallel python process. This happens in the background using a combination of the [zero message queue](https://zeromq.org)
and [cloudpickle](https://github.com/cloudpipe/cloudpickle) to communicate binary python objects. The `pympipool.SocketInterface`
is an abstraction of this interface, which is used in the other classes inside `pympipool` and might also be helpful for
other projects. 