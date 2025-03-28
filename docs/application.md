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
