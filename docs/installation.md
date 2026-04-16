# Installation
## Minimal
Executorlib internally uses the [zero message queue (zmq)](https://zeromq.org) for communication between the Python 
processes and [cloudpickle](https://github.com/cloudpipe/cloudpickle) for serialization of Python functions to communicate
them from one process to another. So for a minimal installation of executorlib only these two dependencies are installed:
```
pip install executorlib
```
Alternative to the [Python package manager](https://pypi.org/project/executorlib/), executorlib can also be installed 
via the [conda package manager](https://anaconda.org/conda-forge/executorlib):
```
conda install -c conda-forge executorlib
```
A number of features are not available in this minimalistic installation of executorlib, these include the execution of 
MPI parallel Python funtions, which requires the [mpi4py](https://mpi4py.readthedocs.io) package, the caching based on 
the hierarchical data format (HDF5), which requires the [h5py](https://www.h5py.org) package, the submission to job 
schedulers, which requires the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) and the 
visualisation of dependencies, which requires a number of visualisation packages. 

## MPI support
The submission of MPI parallel Python functions requires the installation of the [mpi4py](https://mpi4py.readthedocs.io) 
package. This can be installed in combination with executorlib using either the [Python package manager](https://pypi.org/project/mpi4py/):
```
pip install executorlib[mpi]
```
Or alternatively using the [conda package manager](https://anaconda.org/conda-forge/mpi4py):
```
conda install -c conda-forge executorlib mpi4py
```
Given the C++ bindings included in the [mpi4py](https://mpi4py.readthedocs.io) package it is recommended to use a binary
distribution of [mpi4py](https://mpi4py.readthedocs.io) and only compile it manually when a specific version of MPI is 
used. The mpi4py documentation covers the [installation of mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) 
in more detail. 

## Caching 
While the caching is an optional feature for [Single Node Executor](https://executorlib.readthedocs.io/en/latest/1-single-node.html) and 
for the distribution of Python functions in a given allocation of an HPC job scheduler [HPC Job Executors](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html), 
it is required for the submission of individual functions to an HPC job scheduler [HPC Cluster Executors](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html). 
This is required as in [HPC Cluster Executors](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) the 
Python function is stored on the file system until the requested computing resources become available. The caching is 
implemented based on the hierarchical data format (HDF5). The corresponding [h5py](https://www.h5py.org) package can be 
installed using either the [Python package manager](https://pypi.org/project/h5py/):
```
pip install executorlib[cache]
```
Or alternatively using the [conda package manager](https://anaconda.org/conda-forge/h5py):
```
conda install -c conda-forge executorlib h5py
```
Again, given the C++ bindings of the [h5py](https://www.h5py.org) package to the HDF5 format, a binary distribution is 
recommended. The h5py documentation covers the [installation of h5py](https://docs.h5py.org/en/latest/build.html) in 
more detail. 

## HPC Cluster Executor
[HPC Cluster Executor](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html) requires the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) to 
interface with the job schedulers and [h5py](https://www.h5py.org) package to enable caching, as explained above. Both 
can be installed via the [Python package manager](https://pypi.org/project/pysqa/):
```
pip install executorlib[cluster]
```
Or alternatively using the [conda package manager](https://anaconda.org/conda-forge/pysqa):
```
conda install -c conda-forge executorlib h5py pysqa
```
Depending on the choice of job scheduler the [pysqa](https://pysqa.readthedocs.io) package might require additional 
dependencies, still at least for [SLURM](https://slurm.schedmd.com) no additional requirements are needed. The pysqa 
documentation covers the [installation of pysqa](https://pysqa.readthedocs.io/en/latest/installation.html) in more 
detail.

## HPC Job Executor
For optimal performance the [HPC Job Executor](https://executorlib.readthedocs.io/en/latest/3-hpc-job.html) leverages the
[flux framework](https://flux-framework.org) as its recommended job scheduler.

For detailed instructions on configuring the [flux framework](https://flux-framework.org) for different GPU architectures and Jupyter integration, please refer to the [Flux Framework Integration](flux.md) section.

## Visualisation
The visualisation of the dependency graph with the `plot_dependency_graph` parameter requires [pygraphviz](https://pygraphviz.github.io/documentation/stable/). 
This can installed via the [Python package manager](https://pypi.org/project/pygraphviz/):
```
pip install executorlib[graph]
```
Or alternatively using the [conda package manager](https://anaconda.org/conda-forge/pygraphviz):
```
conda install -c conda-forge executorlib pygraphviz matplotlib networkx ipython
```
Again given the C++ bindings of [pygraphviz](https://pygraphviz.github.io/documentation/stable/) to the graphviz library
it is recommended to install a binary distribution. The pygraphviz documentation covers the [installation of pygraphviz](https://pygraphviz.github.io/documentation/stable/install.html) 
in more detail. Furthermore, [matplotlib](https://matplotlib.org), [networkx](https://networkx.org) and [ipython](https://ipython.readthedocs.io) 
are installed as additional requirements for the visualisation.

## For Developers
To install a specific development branch of executorlib you use the [Python package manager](https://pypi.org/project/executorlib/)
and directly install from the Github repository executorlib is hosted on:
```
pip install git+https://github.com/pyiron/executorlib.git@main
```
In this example the `main` branch is selected. To select a different branch just replace `main` with your target branch.