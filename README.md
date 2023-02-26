# pympipool
Scale functions over multiple compute nodes using mpi4py

## Functionality
Write a python test file like `pool.py`: 
```python
import numpy as np
from pympipool import Pool

def calc(i):
    return np.array(i ** 2)

with Pool(cores=2) as p:
    print(p.map(function=calc, lst=[1, 2, 3, 4]))
```

You can execute the python file `pool.py` in a serial python process: 
```
python pool.py
>>> [array(1), array(4), array(9), array(16)]
```
Internally `pympipool` uses `mpi4py` to distribute the four calculation to two processors `cores=2`.

## Installation
As `pympipool` requires `openmpi` and `mpi4py` it is highly recommended to install it via conda: 
```
conda install -c conda-forge pympipool
```
Alternatively, it is also possible to `pympipool` via `pip`: 
```
pip install pympipool
```

## Changelog
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
