# pympipool
[![Unittests](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml/badge.svg)](https://github.com/pyiron/pympipool/actions/workflows/unittest-openmpi.yml)
[![Coverage Status](https://coveralls.io/repos/github/pyiron/pympipool/badge.svg?branch=main)](https://coveralls.io/github/pyiron/pympipool?branch=main)

Scale functions over multiple compute nodes using mpi4py

## Functionality
### Serial subtasks 
Write a python test file like `pool.py`: 
```python
import numpy as np
from pympipool import Pool

def calc(i):
    return np.array(i ** 2)

with Pool(cores=2) as p:
    print(p.map(fn=calc, iterables=[1, 2, 3, 4]))
```

You can execute the python file `pool.py` in a serial python process: 
```
python pool.py
>>> [array(1), array(4), array(9), array(16)]
```
Internally `pympipool` uses `mpi4py` to distribute the four calculation to two processors `cores=2`.

### MPI parallel subtasks
In addition, the individual python functions can also use multiple MPI ranks. Example `ranks.py`:
```python
from pympipool import Pool

def calc(i, comm):
    return i, comm.Get_size(), comm.Get_rank()

with Pool(cores=4, cores_per_task=2) as p:
    print(p.map(fn=calc, iterables=[1, 2, 3, 4]))
```

Here the user-defined function `calc()` receives an additional input parameter `comm` which represents the 
MPI communicator. It can be used just like any other `mpi4py.COMM` object. Here just the size `Get_size()` 
and the rank `Get_rank()` are returned. 

### Futures Interface
In additions to the `map()` function `pympipool` also implements the `concurrent.futures` interface. As the 
tasks are executed in a parallel subprocess using mpi4py, an additional call to the update function `update()` 
is required. Example `submit.py`:  
```python
import numpy as np
from time import sleep
from pympipool import Pool

def calc(i):
    return np.array(i ** 2)

with Pool(cores=2) as p:
    futures = [p.submit(calc, i=i) for i in [1, 2, 3, 4]]
    print([f.done() for f in futures])
    sleep(1)
    p.update()
    print([f.result() for f in futures if f.done()])
```
After the submission using the submit function `submit()` the futures objects are not completed `done()`. Following,
a short call of the sleep function the update function `update()` synchronizes the local futures objects. Consequently, 
the future objects are afterward completed `done()` and the results can be received using the results function 
`result()`. The code above results in the following output:
```
python submit.py
>>> [False, False, False, False]
>>> [array(1), array(4), array(9), array(16)]
```

## Installation
As `pympipool` requires `mpi` and `mpi4py` it is highly recommended to install it via conda: 
```
conda install -c conda-forge pympipool
```
Alternatively, it is also possible to `pympipool` via `pip`: 
```
pip install pympipool
```

## Changelog
### 0.4.0 
* Update test coverage calculation.
* Add `flux-framework` integration.
* Change interface to be compatible to `concurrent.futures.Executor` - not backwards compatible.

### 0.3.0
* Support subtasks with multiple MPI ranks. 
* Close communication socket when closing the `pympipool.Pool`.
* `tqdm` compatibility is updated from `4.64.1` to `4.65.0`
* `pyzmq` compatibility is updated from `25.0.0` to `25.0.2`

### 0.2.0
* Communicate via zmq rather than `stdin` and `stdout`, this enables support for `mpich` and `openmpi`.
* Add error handling to propagate the `Exception`, when it is raised by mapping the function to the arguments.

### 0.1.0
* Major switch of the communication interface between the serial python process and the mpi parallel python process. 
  Previously, functions were converted to source code using `inspect.getsource()` and `dill` was used to convert the
  sourcecode to an binary blob which could then be transferred between the processes. In the new version, the function
  is directly pickled using `cloudpickle` as `cloudpickle` supports both pickle by reference and pickle by value. Here
  the pickle by value functionality is used to pickle the functions which is be communicated.
* The documentation is updated to reflect the changes in the updated version. 

### 0.0.2
* output of the function which is mapped to the arguments is suppressed, as `stdout` interferes with the communication
  of `pympipool`. Consequently, the output of `print` statements is no longer visible. 
* support for python 3.11 is added  
* `mpi4py` compatibility is updated from `3.1.3` to `3.1.4`
* `dill` compatibility is updated from `0.3.5.1` to `0.3.6`
* `tqdm` compatibility is updated from `4.64.0` to `4.64.1`

### 0.0.1
* initial release
