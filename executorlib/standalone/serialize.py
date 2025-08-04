import hashlib
import inspect
import re
from typing import Callable, Optional

import cloudpickle


def cloudpickle_register(ind: int = 2):
    """
    Cloudpickle can either pickle by value or pickle by reference. The functions which are communicated have to
    be pickled by value rather than by reference, so the module which calls the map function is pickled by value.
    https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
    inspect can help to find the module which is calling executorlib
    https://docs.python.org/3/library/inspect.html
    to learn more about inspect another good read is:
    http://pymotw.com/2/inspect/index.html#module-inspect
    1 refers to 1 level higher than the map function

    Args:
        ind (int): index of the level at which pickle by value starts while for the rest pickle by reference is used
    """
    try:  # When executed in a jupyter notebook this can cause a ValueError - in this case we just ignore it.
        cloudpickle.register_pickle_by_value(inspect.getmodule(inspect.stack()[ind][0]))
    except IndexError:
        cloudpickle_register(ind=ind - 1)
    except ValueError:
        pass


def serialize_funct(
    fn: Callable,
    fn_args: Optional[list] = None,
    fn_kwargs: Optional[dict] = None,
    resource_dict: Optional[dict] = None,
    cache_key: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Serialize a function and its arguments and keyword arguments into an HDF5 file.

    Args:
        fn (Callable): The function to be serialized.
        fn_args (list): The arguments of the function.
        fn_kwargs (dict): The keyword arguments of the function.
        resource_dict (dict): A dictionary of resources required by the task. With the following keys:
                              - cores (int): number of MPI cores to be used for each function call
                              - threads_per_core (int): number of OpenMP threads to be used for each function call
                              - gpus_per_core (int): number of GPUs per worker - defaults to 0
                              - cwd (str/None): current working directory where the parallel python task is executed
                              - openmpi_oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI and
                                                              SLURM only) - default False
                              - slurm_cmd_args (list): Additional command line arguments for the srun call (SLURM only)
                              - error_log_file (str): Name of the error log file to use for storing exceptions raised
                                                      by the Python functions submitted to the Executor.
        cache_key (str, optional): By default the cache_key is generated based on the function hash, this can be
                                   overwritten by setting the cache_key.

    Returns:
        Tuple[str, dict]: A tuple containing the task key and the serialized data.

    """
    if fn_args is None:
        fn_args = []
    if fn_kwargs is None:
        fn_kwargs = {}
    if resource_dict is None:
        resource_dict = {}
    if cache_key is not None:
        task_key = cache_key
    else:
        binary_all = cloudpickle.dumps(
            {
                "fn": fn,
                "args": fn_args,
                "kwargs": fn_kwargs,
            }
        )
        task_key = fn.__name__ + _get_hash(binary=binary_all)
    data = {
        "fn": fn,
        "args": fn_args,
        "kwargs": fn_kwargs,
        "resource_dict": resource_dict,
    }
    return task_key, data


def _get_hash(binary: bytes) -> str:
    """
    Get the hash of a binary.

    Args:
        binary (bytes): The binary to be hashed.

    Returns:
        str: The hash of the binary.

    """
    # Remove specification of jupyter kernel from hash to be deterministic
    binary_no_ipykernel = re.sub(b"(?<=/ipykernel_)(.*)(?=/)", b"", binary)
    return str(hashlib.md5(binary_no_ipykernel).hexdigest())
