import os
from typing import Any

from executorlib.cache.shared import FutureItem
from executorlib.standalone.hdf import dump, load


def backend_load_file(file_name: str) -> dict:
    """
    Load the data from an HDF5 file and convert FutureItem objects to their results.

    Args:
        file_name (str): The name of the HDF5 file.

    Returns:
        dict: The loaded data from the file.

    """
    apply_dict = load(file_name=file_name)
    apply_dict["args"] = [
        arg if not _isinstance(arg, FutureItem) else arg.result()
        for arg in apply_dict["args"]
    ]
    apply_dict["kwargs"] = {
        key: arg if not _isinstance(arg, FutureItem) else arg.result()
        for key, arg in apply_dict["kwargs"].items()
    }
    return apply_dict


def backend_write_file(file_name: str, output: Any) -> None:
    """
    Write the output to an HDF5 file.

    Args:
        file_name (str): The name of the HDF5 file.
        output (Any): The output to be written.

    Returns:
        None

    """
    file_name_out = os.path.splitext(file_name)[0]
    os.rename(file_name, file_name_out + ".h5ready")
    dump(file_name=file_name_out + ".h5ready", data_dict={"output": output})
    os.rename(file_name_out + ".h5ready", file_name_out + ".h5out")


def backend_execute_task_in_file(file_name: str) -> None:
    """
    Execute the task stored in a given HDF5 file.

    Args:
        file_name (str): The file name of the HDF5 file as an absolute path.

    Returns:
        None
    """
    apply_dict = backend_load_file(file_name=file_name)
    result = apply_dict["fn"].__call__(*apply_dict["args"], **apply_dict["kwargs"])
    backend_write_file(
        file_name=file_name,
        output=result,
    )


def _isinstance(obj: Any, cls: type) -> bool:
    return str(obj.__class__) == str(cls)
