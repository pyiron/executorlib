import hashlib
import inspect
import re
from typing import Any, Tuple

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


def serialize_funct_h5(
    fn: callable, fn_args: list = [], fn_kwargs: dict = {}, resource_dict: dict = {}
) -> Tuple[str, dict]:
    """
    Serialize a function and its arguments and keyword arguments into an HDF5 file.

    Args:
        fn (callable): The function to be serialized.
        fn_args (list): The arguments of the function.
        fn_kwargs (dict): The keyword arguments of the function.
        resource_dict (dict): resource dictionary, which defines the resources used for the execution of the function.
                              Example resource dictionary: {
                                  cores: 1,
                                  threads_per_core: 1,
                                  gpus_per_worker: 0,
                                  oversubscribe: False,
                                  cwd: None,
                                  executor: None,
                                  hostname_localhost: False,
                              }

    Returns:
        Tuple[str, dict]: A tuple containing the task key and the serialized data.

    """
    binary_all = cloudpickle.dumps(
        {"fn": fn, "args": fn_args, "kwargs": fn_kwargs, "resource_dict": resource_dict}
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
