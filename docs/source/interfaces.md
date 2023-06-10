# Interfaces
## Pool
`pympipool.Pool`: Following the [`multiprocessing.pool.Pool`](https://docs.python.org/3/library/multiprocessing.html) 
the `pympipool.Pool` class implements the `map()` and `starmap()` functions. Internally these connect to an MPI parallel
subprocess running the [`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
So by increasing the number of workers, by setting the `max_workers` parameter the `pympipool.Pool` can scale the 
execution of serial python functions beyond a single compute node. For MPI parallel python functions the `pympipool.MPISpawnPool`
is derived from the `pympipool.Pool` and uses `MPI_Spawn()` to execute those.  

### Example 
Write a python test file like `test_pool.py`: 
```python
import numpy as np
from pympipool import Pool
def calc(i):
    return np.array(i ** 2)
with Pool(cores=2) as p:
    print(p.map(func=calc, iterable=[1, 2, 3, 4]))
```

You can execute the python file `test_pool.py` in a serial python process: 
```
python test_pool.py
>>> [array(1), array(4), array(9), array(16)]
```
Internally `pympipool` uses `mpi4py` to distribute the four calculation to two processors `cores=2`.

## Executor
`pympipool.Executor`: The easiest way to execute MPI parallel python functions right next to serial python functions 
is the `pympipool.Executor`. It implements the executor interface defined by the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures).
So functions are submitted to the `pympipool.Executor` using the `submit()` function, which returns an [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
object. With these [`concurrent.futures.Future`](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
objects asynchronous workflows can constructed which periodically check if the computation is completed `done()` and then
query the results using the `result()` function. The limitation of the `pympipool.Executor` is lack of load balancing, 
each `pympipool.Executor` acts as a serial first in first out (FIFO) queue. So it is the task of the user to balance the
load of many different tasks over multiple `pympipool.Executor` instances. 

## PoolExecutor
`pympipool.PoolExecutor`: To combine the functionality of the `pympipool.Pool` and the `pympipool.Executor` the 
`pympipool.PoolExecutor` again connects to the [`mpi4py.futures.MPIPoolExecutor`](https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor).
Still in contrast to the `pympipool.Pool` it does not implement the `map()` and `starmap()` functions but rather the 
`submit()` function based on the [`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures)
interface. In this case the load balancing happens internally and the maximum number of workers `max_workers` defines
the maximum number of parallel tasks. But only serial python tasks can be executed in contrast to the `pympipool.Executor`
which can also execute MPI parallel python tasks. 

## MPISpawnPool
`pympipool.MPISpawnPool`: An alternative way to support MPI parallel functions in addition to the `pympipool.Executor`
is the `pympipool.MPISpawnPool`. Just like the `pympipool.Pool` it supports the `map()` and `starmap()` functions. The 
additional `ranks_per_task` parameter defines how many MPI ranks are used per task. All functions are executed with the
same number of MPI ranks. The limitation of this approach is that it uses `MPI_Spawn()` to create new MPI ranks for the
execution of the individual tasks. Consequently, this approach is not as scalable as the `pympipool.Executor` but it 
offers load balancing for a large number of similar MPI parallel tasks. 

### Example 
Write a python test file like `test_mpispawnpool.py`: 
```python
from pympipool import MPISpawnPool
def calc(i, comm):
    return i, comm.Get_size(), comm.Get_rank()
with MPISpawnPool(cores=4, cores_per_task=2) as p:
    print(p.map(func=calc, iterable=[1, 2, 3, 4]))
```

You can execute the python file `test_mpispawnpool.py` in a serial python process: 
```
python test_mpispawnpool.py
>>> [array(1), array(4), array(9), array(16)]
```
Internally `pympipool` uses `mpi4py` to distribute the four calculation to two processors `cores=2`.

## SocketInterface
`pympipool.SocketInterface`: The key functionality of the `pympipool` package is the coupling of a serial python process
with an MPI parallel python process. This happens in the background using a combination of the [zero message queue](https://zeromq.org)
and [cloudpickle](https://github.com/cloudpipe/cloudpickle) to communicate binary python objects. The `pympipool.SocketInterface`
is an abstraction of this interface, which is used in the other classes inside `pympipool` and might also be helpful for
other projects. 