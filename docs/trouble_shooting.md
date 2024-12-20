# Trouble Shooting
Some of the most frequent issues are covered below, for everything else do not be shy and [open an issue on Github](https://github.com/pyiron/executorlib/issues).

## Filesystem Usage
The cache of executorlib is not removed after the Python process completed. So it is the responsibility of the user to 
clean up the cache directory they created. This can be easily forgot, so it is important to check for remaining cache 
directories from time to time and remove them. 

## Firewall Issues
MacOS comes with a rather strict firewall, which does not allow to connect to an MacOS computer using the hostname even
if it is the hostname of the current computer. MacOS only supports connections based on the hostname `localhost`. To use
`localhost` rather than the hostname to connect to the Python processes executorlib uses for the execution of the Python
function, executorlib provides the option to set `hostname_localhost=True`. For MacOS this option is enabled by default,
still if other operating systems implement similar strict firewall rules, the option can also be set manually to enabled
local mode on computers with strict firewall rules.

## Message Passing Interface
To use the message passing interface (MPI) executorlib requires [mpi4py](https://mpi4py.readthedocs.io/) as optional 
dependency. The installation of this and other optional dependencies is covered in the [installation section](https://executorlib.readthedocs.io/en/latest/installation.html#mpi-support).

## Missing Dependencies
The default installation of executorlib only comes with a limited number of dependencies, especially the [zero message queue](https://zeromq.org)
and [cloudpickle](https://github.com/cloudpipe/cloudpickle). Additional features like [caching](https://executorlib.readthedocs.io/en/latest/installation.html#caching), [HPC submission mode](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-submission-mode) 
and [HPC allocation mode](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-allocation-mode) require additional dependencies. The dependencies are explained in more detail in the 
[installation section](https://executorlib.readthedocs.io/en/latest/installation.html#).

## Python Version 
Executorlib supports all current Python version ranging from 3.9 to 3.13. Still some of the dependencies and especially 
the [flux](http://flux-framework.org) job scheduler are currently limited to Python 3.12 and below. Consequently for high
performance computing installations Python 3.12 is the recommended Python verion. 

## Resource Dictionary
The resource dictionary parameter `resource_dict` can contain one or more of the following options: 
* `cores` (int): number of MPI cores to be used for each function call
* `threads_per_core` (int): number of OpenMP threads to be used for each function call
* `gpus_per_core` (int): number of GPUs per worker - defaults to 0
* `cwd` (str/None): current working directory where the parallel python task is executed
* `openmpi_oversubscribe` (bool): adds the `--oversubscribe` command line flag (OpenMPI and SLURM only) - default False
* `slurm_cmd_args` (list): Additional command line arguments for the srun call (SLURM only)

For the special case of the [HPC allocation mode](https://executorlib.readthedocs.io/en/latest/3-hpc-allocation.html) 
the resource dictionary parameter `resource_dict` can also include additional parameters define in the submission script
of the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) these include but are not limited to: 
* `run_time_max` (int): the maximum time the execution of the submitted Python function is allowed to take in seconds.
* `memory_max` (int): the maximum amount of memory the Python function is allowed to use in Gigabytes. 
* `partition` (str): the partition of the queuing system the Python function is submitted to. 
* `queue` (str): the name of the queue the Python function is submitted to. 

All parameters in the resource dictionary `resource_dict` are optional. 

## SSH Connection
While the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) provides the option to connect to
high performance computing (HPC) clusters via SSH, this functionality is not supported for executorlib. The background 
is the use of [cloudpickle](https://github.com/cloudpipe/cloudpickle) for serialization inside executorlib, this requires
the same Python version and dependencies on both computer connected via SSH. As tracking those parameters is rather 
complicated the SSH connection functionality of [pysqa](https://pysqa.readthedocs.io) is not officially supported in 
executorlib. 
