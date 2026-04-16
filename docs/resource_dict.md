# Resource Dictionary
The resource dictionary parameter `resource_dict` is used to specify the computing resources allocated to the execution of a submitted Python function. This flexibility allows users to assign resources on a per-function-call basis, simplifying the up-scaling of Python programs.

## Available Options
The `resource_dict` can contain one or more of the following options:

* **`cores`** (int): Number of MPI cores to be used for each function call.
* **`threads_per_core`** (int): Number of OpenMP threads to be used for each function call.
* **`gpus_per_core`** (int): Number of GPUs per worker - defaults to 0.
* **`cwd`** (str/None): Current working directory where the parallel python task is executed.
* **`cache_key`** (str): Rather than using the internal hashing of executorlib, the user can provide an external `cache_key` to identify tasks on the file system. The initial file name will be `cache_key + "_i.h5"` and the final file name will be `cache_key + "_o.h5"`.
* **`cache_directory`** (str): The directory to store cache files.
* **`num_nodes`** (int): Number of compute nodes used for the evaluation of the Python function.
* **`exclusive`** (bool): Boolean flag to reserve exclusive access to selected compute nodes - do not allow other tasks to use the same compute node.
* **`error_log_file`** (str): Path to the error log file, primarily used to merge the log of multiple tasks in one file.
* **`run_time_max`** (int): The maximum time the execution of the submitted Python function is allowed to take in seconds.
* **`priority`** (int): The queuing system priority assigned to a given Python function to influence the scheduling.
* **`slurm_cmd_args`** (list): Additional command line arguments for the `srun` call (SLURM only).

## HPC Job Executor Specifics
For the special case of the [HPC Job Executor](3-hpc-job.ipynb), the `resource_dict` can also include additional parameters defined in the submission script of the [Python simple queuing system adapter (pysqa)](https://pysqa.readthedocs.io). These include but are not limited to:

* **`memory_max`** (int): The maximum amount of memory the Python function is allowed to use in Gigabytes.
* **`partition`** (str): The partition of the queuing system the Python function is submitted to.
* **`queue`** (str): The name of the queue the Python function is submitted to.

## Validation
All parameters in the `resource_dict` are optional. When `pydantic` is installed as an optional dependency, the `resource_dict` is automatically validated using `pydantic`.
