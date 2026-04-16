# Flux Framework Integration
For optimal performance the [HPC Job Executor](3-hpc-job.ipynb) leverages the
[flux framework](https://flux-framework.org) as its recommended job scheduler. Even when the [Simple Linux Utility for Resource Management (SLURM)](https://slurm.schedmd.com)
or any other job scheduler is already installed on the HPC cluster, the [flux framework](https://flux-framework.org) can be
installed as a secondary job scheduler to leverage [flux framework](https://flux-framework.org) for the distribution of
resources within a given allocation of the primary scheduler.

The [flux framework](https://flux-framework.org) uses `libhwloc` and `pmi` to understand the hardware it is running on
and to bootstrap MPI. `libhwloc` not only assigns CPU cores but also GPUs. This requires `libhwloc` to be compiled with
support for GPUs from your vendor. In the same way the version of `pmi` for your queuing system has to be compatible
with the version installed via conda. As `pmi` is typically distributed with the implementation of the Message Passing
Interface (MPI), it is required to install the compatible MPI library in your conda environment as well.

## GPU Support
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
conda install -c conda-forge flux-core flux-sched mpich>=4 executorlib
```

## Advanced Configuration
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
In addition, the `pmi_mode="pmix"` parameter has to be set for the `FluxJobExecutor` or the
`FluxClusterExecutor` to switch to `pmix` as backend.

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
srun --mpi=pmi2 flux start python <script.py>
```

## Jupyter Integration
### Flux with Jupyter
Two options are available to use flux inside the jupyter notebook or jupyter lab environment. The first option is to
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
