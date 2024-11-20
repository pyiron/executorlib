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
While the caching is an optional feature for [Local Mode] and for the distribution of Python functions in a given 
allocation of an HPC job scheduler [HPC Allocation Mode], it is required for the submission of individual functions to
an HPC job scheduler [HPC Submission Mode]. This is required as in [HPC Submission Mode] the Python function is stored
on the file system until the requested computing resources become available. The caching is implemented based on the 
hierarchical data format (HDF5). The corresponding [h5py](https://www.h5py.org) package can be installed using either 
the [Python package manager](https://pypi.org/project/h5py/):
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

## HPC Submission Mode
[HPC Submission Mode] requires the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) to 
interface with the job schedulers and [h5py](https://www.h5py.org) package to enable caching, as explained above. Both 
can be installed via the [Python package manager](https://pypi.org/project/pysqa/):
```
pip install executorlib[submission]
```
Or alternatively using the [conda package manager](https://anaconda.org/conda-forge/pysqa):
```
conda install -c conda-forge executorlib h5py pysqa
```
Depending on the choice of job scheduler the [pysqa](https://pysqa.readthedocs.io) package might require additional 
dependencies, still at least for [SLURM](https://slurm.schedmd.com) no additional requirements are needed. The pysqa 
documentation covers the [installation of pysqa](https://pysqa.readthedocs.io/en/latest/installation.html) in more 
detail.

## HPC Allocation Mode
For optimal performance in [HPC Allocation Mode] the [flux framework](https://flux-framework.org) is recommended as job
scheduler. Even when the [Simple Linux Utility for Resource Management (SLURM)](https://slurm.schedmd.com) or any other 
job scheduler is already installed on the HPC cluster. [flux framework](https://flux-framework.org) can be installed as
a secondary job scheduler to leverage [flux framework](https://flux-framework.org) for the distribution of resources 
within a given allocation of the primary scheduler. 

The [flux framework](https://flux-framework.org) uses `libhwloc` and `pmi` to understand the hardware it is running on and to booststrap MPI.
`libhwloc` not only assigns CPU cores but also GPUs. This requires `libhwloc` to be compiled with support for GPUs from 
your vendor. In the same way the version of `pmi` for your queuing system has to be compatible with the version 
installed via conda. As `pmi` is typically distributed with the implementation of the Message Passing Interface (MPI), 
it is required to install the compatible MPI library in your conda environment as well. 

### AMD GPUs with mpich / cray mpi
For example the [Frontier HPC](https://www.olcf.ornl.gov/frontier/) cluster at Oak Ridge National Laboratory uses 
AMD MI250X GPUs with cray mpi version which is compatible to mpich `4.X`. So the corresponding versions can be installed
from conda-forge using: 
```
conda install -c conda-forge flux-core flux-sched libhwloc=*=rocm* mpich>=4 executorlib
```
### Nvidia GPUs with mpich / cray mpi 
For example the [Perlmutter HPC](https://docs.nersc.gov/systems/perlmutter/) at the National Energy Research Scientific 
Computing (NERSC) uses Nvidia A100 GPUs in combination with cray mpi which is compatible to mpich `4.X`. So the 
corresponding versions can be installed from conda-forge using: 
```
conda install -c conda-forge flux-core flux-sched libhwloc=*=cuda* mpich>=4 executorlib
```
When installing on a login node without a GPU the conda install command might fail with an Nvidia cuda related error, in
this case adding the environment variable:
```
CONDA_OVERRIDE_CUDA="11.6"
```
With the specific Nvidia cuda library version installed on the cluster enables the installation even when no GPU is 
present on the computer used for installing. 

### Intel GPUs with mpich / cray mpi 
For example the [Aurora HPC](https://www.alcf.anl.gov/aurora) cluster at Argonne National Laboratory uses Intel Ponte 
Vecchio GPUs in combination with cray mpi which is compatible to mpich `4.X`. So the corresponding versions can be 
installed from conda-forge using: 
```
conda install -c conda-forge flux-core flux-sched mpich=>4 executorlib
```

### Alternative Installations
Flux is not limited to mpich / cray mpi, it can also be installed in compatibility with openmpi or intel mpi using the 
openmpi package: 
```
conda install -c conda-forge flux-core flux-sched openmpi=4.1.6 executorlib
```
For the version 5 of openmpi the backend changed to `pmix`, this requires the additional `flux-pmix` plugin:
```
conda install -c conda-forge flux-core flux-sched flux-pmix openmpi>=5 executorlib
```
In addition, the `flux_executor_pmi_mode="pmix"` parameter has to be set for the `executorlib.Executor` to switch to 
`pmix` as backend.

### Test Flux Framework
To validate the installation of flux and confirm the GPUs are correctly recognized, you can start a flux session on the 
login node using:
```
flux start
```
This returns an interactive shell which is connected to the flux scheduler. In this interactive shell you can now list 
the available resources using: 
```
flux resource list
```
The output should return a list comparable to the following example output:
```
     STATE NNODES   NCORES    NGPUS NODELIST
      free      1        6        1 ljubi
 allocated      0        0        0 
      down      0        0        0
```
As flux only lists physical cores rather than virtual cores enabled by hyper-threading the total number of CPU cores 
might be half the number of cores you expect. 

### Flux Framework as Secondary Scheduler
When the flux framework is used inside an existing queuing system, you have to communicate the available resources to 
the flux framework. For SLURM this is achieved by calling `flux start` with `srun`. For an interactive session use:
```
srun --pty flux start
```
Alternatively, to execute a python script `<script.py>` which uses `executorlib` you can call it with: 
```
srun flux start python <script.py>
```

### PMI Compatibility 
When pmi version 1 is used rather than pmi version 2 then it is possible to enforce the usage of `pmi-2` during the 
startup process of flux using: 
```
srun –mpi=pmi2 flux start python <script.py>
```

### Flux with Jupyter 
To options are available to use flux inside the jupyter notebook or jupyter lab environment. The first option is to
start the flux session and then start the jupyter notebook inside the flux session. This just requires a single call on 
the command line:
```
flux start jupyter notebook
```
The second option is to create a separate Jupyter kernel for flux. This option requires multiple steps of configuration, 
still it has the advantage that it is also compatible with the multi-user jupyterhub environment. Start by identifying 
the directory Jupyter searches for Jupyter kernels: 
```
jupyter kernelspec list
```
This returns a list of jupyter kernels, commonly stored in `~/.local/share/jupyter`. It is recommended to create the
flux kernel in this directory. Start by creating the corresponding directory by copying one of the existing kernels:
```
cp -r ~/.local/share/jupyter/kernels/python3 ~/.local/share/jupyter/kernels/flux
```
In the directory a JSON file is created which contains the configuration of the Jupyter Kernel. You can use an editor of
your choice, here we use vi to create the `kernel.json` file:
```
vi ~/.local/share/jupyter/kernels/flux/kernel.json
```
Inside the file copy the following content. The first entry under the name `argv` provides the command to start the 
jupyter kernel. Typically this would be just calling python with the parameters to launch an ipykernel. In front of this
command the `flux start` command is added. 
```
{
  "argv": [
    "flux",
    "start",
    "/srv/conda/envs/notebook/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Flux",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
```
More details for the configuration of Jupyter kernels is available as part of the [Jupyter documentation](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernel-specs).

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