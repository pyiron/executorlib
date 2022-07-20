# pympipool
Scale functions over multiple compute nodes using mpi4py

```python
from pympipool import Pool

def calc(i):
    import numpy as np
    return np.array(i ** 2)

with Pool(cores=2) as p:
    print(p.map(function=calc, lst=[1, 2, 3, 4]))

>>> [array(1), array(4), array(9), array(16)]
```