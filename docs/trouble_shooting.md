# Trouble Shooting
Some of the most frequent issues are covered below, for everything else do not be shy and [open an issue on Github](https://github.com/pyiron/executorlib/issues).

## Filesystem Usage
The cache of executorlib is not removed after the Python process completed. So it is the responsibility of the user to 
clean up the cache directory they created. This can be easily forgot, so it is important to check for remaining cache 
directories from time to time and remove them. In addition, there is no guarantee for cache compatibility between 
different versions, the cache is only intended for temporary use and it is not designed for long-term storage. 

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
and [cloudpickle](https://github.com/cloudpipe/cloudpickle). Additional features like [caching](https://executorlib.readthedocs.io/en/latest/installation.html#caching), the [HPC Cluster Executors](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-cluster-executor) 
and the [HPC Job Executors](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-job-executor) require 
additional dependencies. The dependencies are explained in more detail in the 
[installation section](https://executorlib.readthedocs.io/en/latest/installation.html).

Typical error messages related to missing dependencies are `ModuleNotFoundError` like the following:
* `ModuleNotFoundError: No module named 'pysqa'` - Install [pysqa](https://pysqa.readthedocs.io/) as explained in the [HPC Cluster Executors](https://executorlib.readthedocs.io/en/latest/installation.html#hpc-cluster-executor) section of the installation.
* `ModuleNotFoundError: No module named 'h5py'` - Install [h5py](https://www.h5py.org/) as explained in the [Caching](https://executorlib.readthedocs.io/en/latest/installation.html#caching) section of the installation. 
* `ModuleNotFoundError: No module named 'networkx'` - Install [networkx](https://networkx.org/) as explained in the [Visualisation](https://executorlib.readthedocs.io/en/latest/installation.html#visualisation) section of the installation.

## Test Coverage for Integration Tests
When Python functions are executed with executorlib, they run in subprocesses started via `sys.executable`. As a result, `coverage run`
only tracks the main test process by default and can miss function execution inside executorlib workers.

To collect coverage from both the main process and executorlib subprocesses, enable the subprocess patch in your project configuration:
```toml
[tool.coverage.run]
patch = ["subprocess"]
```

Then execute:
```bash
coverage run -m unittest discover
coverage combine
coverage report
```

The `coverage combine` command merges the data from the main process and subprocesses.

## Python Version 
Executorlib supports all current Python version ranging from 3.9 to 3.13. Still some of the dependencies and especially 
the [flux](http://flux-framework.org) job scheduler are currently limited to Python 3.12 and below. Consequently for high
performance computing installations Python 3.12 is the recommended Python verion. 

## Resource Dictionary
The `resource_dict` parameter is a central part of `executorlib` to assign computing resources on a per-function-call basis. For a complete list of available options and their descriptions, please refer to the [Resource Dictionary](resource_dict.md) section.

## SSH Connection
While the [Python simple queuing system adatper (pysqa)](https://pysqa.readthedocs.io) provides the option to connect to
high performance computing (HPC) clusters via SSH, this functionality is not supported for executorlib. The background 
is the use of [cloudpickle](https://github.com/cloudpipe/cloudpickle) for serialization inside executorlib, this requires
the same Python version and dependencies on both computer connected via SSH. As tracking those parameters is rather 
complicated the SSH connection functionality of [pysqa](https://pysqa.readthedocs.io) is not officially supported in 
executorlib. 
