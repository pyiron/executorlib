import hashlib
import re
from typing import Any, Tuple

import cloudpickle


def serialize_funct_h5(fn: callable, *args: Any, **kwargs: Any) -> Tuple[str, dict]:
    """
    Serialize a function and its arguments and keyword arguments into an HDF5 file.

    Args:
        fn (callable): The function to be serialized.
        *args (Any): The arguments of the function.
        **kwargs (Any): The keyword arguments of the function.

    Returns:
        Tuple[str, dict]: A tuple containing the task key and the serialized data.

    """
    binary_all = cloudpickle.dumps({"fn": fn, "args": args, "kwargs": kwargs})
    task_key = fn.__name__ + _get_hash(binary=binary_all)
    data = {"fn": fn, "args": args, "kwargs": kwargs}
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
