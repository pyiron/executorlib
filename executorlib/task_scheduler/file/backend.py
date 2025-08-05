import os
import time
from typing import Any

from executorlib.standalone.error import backend_write_error_file
from executorlib.standalone.hdf import dump, load
from executorlib.task_scheduler.file.shared import FutureItem


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


def backend_write_file(file_name: str, output: Any, runtime: float) -> None:
    """
    Write the output to an HDF5 file.

    Args:
        file_name (str): The name of the HDF5 file.
        output (Any): The output to be written.
        runtime (float): Time for executing function.

    Returns:
        None

    """
    file_name_out = os.path.splitext(file_name)[0][:-2]
    os.rename(file_name, file_name_out + "_r.h5")
    if "result" in output:
        dump(
            file_name=file_name_out + "_r.h5",
            data_dict={"output": output["result"], "runtime": runtime},
        )
    else:
        dump(
            file_name=file_name_out + "_r.h5",
            data_dict={"error": output["error"], "runtime": runtime},
        )
    os.rename(file_name_out + "_r.h5", file_name_out + "_o.h5")


def backend_execute_task_in_file(file_name: str) -> None:
    """
    Execute the task stored in a given HDF5 file.

    Args:
        file_name (str): The file name of the HDF5 file as an absolute path.

    Returns:
        None
    """
    apply_dict = backend_load_file(file_name=file_name)
    time_start = time.time()
    try:
        result = {
            "result": apply_dict["fn"].__call__(
                *apply_dict["args"], **apply_dict["kwargs"]
            )
        }
    except Exception as error:
        result = {"error": error}
        backend_write_error_file(
            error=error,
            apply_dict=apply_dict,
        )

    backend_write_file(
        file_name=file_name,
        output=result,
        runtime=time.time() - time_start,
    )


def _isinstance(obj: Any, cls: type) -> bool:
    return str(obj.__class__) == str(cls)
