# Application
While `executorlib` is designed to up-scale any Python function for high performance computing (HPC), it was initially
developed to accelerate atomistic computational materials science simulation. To demonstrate the usage of `executorlib`
in the context of atomistic simulation, it is combined with [atomistics](https://atomistics.readthedocs.io/) and the
[atomic simulation environment (ASE)](https://wiki.fysik.dtu.dk/ase/) to calculate the bulk modulus with two density
functional theory simulation codes [gpaw](https://gpaw.readthedocs.io/index.html) and [quantum espresso](https://www.quantum-espresso.org).
The bulk modulus is calculated by uniformly deforming a supercell of atoms and measuring the change in total energy 
during compression and elongation. The first derivative of this curve is the pressure and the second derivative is 
proportional to the bulk modulus. Other material properties like the heat capacity, thermal expansion or thermal conductivity
can be calculated in similar ways following the [atomistics](https://atomistics.readthedocs.io/) documentation. 

## Pandas DataFrame operations
Beyond atomistic simulations, `executorlib` can also accelerate tabular data processing. For example, replacing
`DataFrame.apply()` with `SingleNodeExecutor.map()` enables parallel evaluation for row-wise operations:

```python
import pandas
from executorlib import SingleNodeExecutor


def compute(a, b):
    return (a - b) / (a + b)


df = pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
with SingleNodeExecutor() as exe:
    df["c"] = list(exe.map(compute, df["a"], df["b"]))
```
