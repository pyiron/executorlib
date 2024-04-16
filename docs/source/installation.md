# Installation
For up-scaling python functions beyond a single compute node `pympipool` requires the integration with a high 
performance computing (HPC) resource manager. These HPC resource manager are currently only supported for Linux. Still
for testing and development purposes the `pympipool` package can installed on all major operating systems including 
Windows. 

This basic installation is based on the `pympipool.mpi.PyMPIExecutor` interface and allows up-scaling serial 
and parallel python functions which use the message passing interface (MPI) for python [`mpi4py`](https://mpi4py.readthedocs.io)
on a single compute node. In addition, the integration with an HPC resource manager provides scaling beyond one compute
node, thread based parallelism and the assignment of GPUs. Still the user would not call the interface directly, but 
rather use it through the `pympipool.Executor`. 

## Basic Installation 
For testing and development purposes the `pympipool` package can installed on all major operating systems including 
Windows. It is recommended to use the [conda package manager](https://anaconda.org/conda-forge/pympipool) for the 
installation of the `pympipool` package. Still for advanced users who aim at maximizing their performance by compiling 
their own version of `mpi` and `mpi4py` the `pympipool` package is also provided via the 
[python package index (pypi)](https://pypi.org/project/pympipool/).

### conda-based installation 
In the same way `pympipool` can be installed with the [conda package manager](https://anaconda.org/conda-forge/pympipool): 
```shell
conda install -c conda-forge pympipool
```
When resolving the dependencies with `conda` gets slow it is recommended to use `mamba` instead of `conda`. So you can 
also install `pympipool` using: 
```shell
mamba install -c conda-forge pympipool
```

### pypi-based installation
`pympipool` can be installed from the [python package index (pypi)](https://pypi.org/project/pympipool/) using the 
following command: 
```shell
pip install pympipool
```

## High Performance Computing
`pympipool` currently provides interfaces to the [SLURM workload manager](https://www.schedmd.com) and the 
[flux framework](https://flux-framework.org). With the [flux framework](https://flux-framework.org) being the 
recommended solution as it can be installed without root permissions and it can be integrated in existing resource
managers like the [SLURM workload manager](https://www.schedmd.com). The advantages of using `pympipool` in combination
with these resource schedulers is the fine-grained resource allocation. In addition to scaling beyond a single compute
node, they add the ability to assign GPUs and thread based parallelism. The two resource manager are internally linked to
two interfaces: 

* `pympipool.slurm.PySlurmExecutor`: The interface for the [SLURM workload manager](https://www.schedmd.com).
* `pympipool.flux.PyFluxExecutor`: The interface for the [flux framework](https://flux-framework.org).

Still the user would not call these interfaces directly, but rather use it through the `pympipool.Executor`. 

### Flux Framework
For Linux users without a pre-installed resource scheduler in their high performance computing (HPC) environment, the
[flux framework](https://flux-framework.org) can be installed with the `conda` package manager: 
```shell
conda install -c conda-forge flux-core
```
For alternative ways to install the [flux framework](https://flux-framework.org) please refer to their official 
[documentation](https://flux-framework.readthedocs.io/en/latest/quickstart.html).

#### Nvidia 
For adding GPU support in the [flux framework](https://flux-framework.org) you want to install `flux-sched` in addition 
to `flux-core`. For Nvidia GPUs you need: 
```shell
conda install -c conda-forge flux-core flux-sched libhwloc=*=cuda*
```
In case this fails because there is no GPU on the login node and the `cudatoolkit` cannot be installed you can use the 
`CONDA_OVERRIDE_CUDA` environment variable to pretend a local cuda version is installed `conda` can link to using:
```shell
CONDA_OVERRIDE_CUDA="11.6" conda install -c conda-forge flux-core flux-sched libhwloc=*=cuda*
```

#### AMD
For adding GPU support in the [flux framework](https://flux-framework.org) you want to install `flux-sched` in addition 
to `flux-core`. For AMD GPUs you need: 
```shell
conda install -c conda-forge flux-core flux-sched
```

#### Test Flux
To test the [flux framework](https://flux-framework.org) and validate the GPUs are correctly recognized you can start
a flux instance using: 
```shell
flux start
```
Afterwards, you can list the resources accessible to flux using:
```shell
flux resource list
```
This should contain a column for the GPUs if you installed the required dependencies. Here is an example output for a 
workstation with a six core CPU and a single GPU: 
```
     STATE NNODES   NCORES    NGPUS NODELIST
      free      1        6        1 ljubi
 allocated      0        0        0 
      down      0        0        0 
```
As the [flux framework](https://flux-framework.org) only lists physical cores rather than virtual cores enabled by
hyper-threading the total number of CPU cores might be half the number of cores you expect.

When the [flux framework](https://flux-framework.org) is used inside an existing queuing system, then you have to 
communicate these resources to it. For the [SLURM workload manager](https://www.schedmd.com) this is achieved by calling
`flux start` with `srun`. For an interactive session use: 
```shell
srun --pty flux start
```
Alternatively, to execute a python script which uses `pympipool` you can call it with: 
```shell
srun flux start python <your python script.py>
```
In the same way to start a Jupyter Notebook in an interactive allocation you can use: 
```shell
srun --pty flux start jupyter notebook
```
Then each jupyter notebook you execute on this jupyter notebook server has access to the resources of the interactive
allocation. 

### SLURM 
The installation of the [SLURM workload manager](https://www.schedmd.com) is explained in the corresponding 
[documentation](https://slurm.schedmd.com/quickstart_admin.html) . As it requires root access, it is not explained here.
Rather we assume you have access to an HPC cluster which already has SLURM installed. 

While the [SLURM workload manager](https://www.schedmd.com) and the [flux framework](https://flux-framework.org) are 
both resource schedulers, the [flux framework](https://flux-framework.org) can also be installed on an HPC system which
uses the [SLURM workload manager](https://www.schedmd.com) as primary resource scheduler. This enables more fine-grained
scheduling like independent GPU access on HPC systems where [SLURM workload manager](https://www.schedmd.com) is 
configured to allow only one job step per node. Furthermore, the [flux framework](https://flux-framework.org) provides 
superior performance in large allocation with several hundred compute nodes or in the case when many `pympipool.slurm.PySlurmExecutor`
objects are created frequently, as each creation of an `pympipool.slurm.PySlurmExecutor` results in an `srun` call which is 
communicated to the central database of the [SLURM workload manager](https://www.schedmd.com). 
