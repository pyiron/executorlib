# pympipool
Scale functions over multiple compute nodes using mpi4py

Write a python test file like `pool.py`: 
```python
from pympipool import Pool

def calc(i):
    import numpy as np
    return np.array(i ** 2)

with Pool(cores=2) as p:
    print(p.map(function=calc, lst=[1, 2, 3, 4]))
```

You can execute the python file `pool.py` in a serial python process: 
```
python pool.py
>>> [array(1), array(4), array(9), array(16)]
```
Internally `pympipool` uses `mpi4py` to distribute the 4 calculation to two processors `cores=2`.  

Alternatively, you can copy all lines to a `jupyter notebook` or an `ipython` shell and execute them directly. But this does not work with a regular python shell. 
