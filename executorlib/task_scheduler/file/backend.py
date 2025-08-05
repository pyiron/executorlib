import os
import time
from typing import Any

from executorlib.standalone.error import backend_write_error_file
from executorlib.standalone.hdf import dump_to_hdf, load_from_hdf
from executorlib.task_scheduler.file.shared import FutureItem


def backend_load_file(file_name: str, load_function: callable = load_from_hdf) -> dict:
    """
    Load the data from an HDF5 file and convert FutureItem objects to their results.

    Args:
        file_name (str): The name of the HDF5 file.
        load_function (callable): function to load data from file with file name file_name

    Returns:
        dict: The loaded data from the file.

    """
    apply_dict = load_function(file_name=file_name)
    apply_dict["args"] = [
        arg if not _isinstance(arg, FutureItem) else arg.result()
        for arg in apply_dict["args"]
    ]
    apply_dict["kwargs"] = {
        key: arg if not _isinstance(arg, FutureItem) else arg.result()
        for key, arg in apply_dict["kwargs"].items()
    }
    return apply_dict


def backend_write_file(
    file_name: str, output: Any, runtime: float, dump_function: callable = dump_to_hdf
) -> None:
    """
    Write the output to an HDF5 file.

    Args:
        file_name (str): The name of the HDF5 file.
        output (Any): The output to be written.
        runtime (float): Time for executing function.
        dump_function (callable): function to dump output to file

    Returns:
        None

    """
    file_name_in, file_extension = os.path.splitext(file_name)
    file_name_out = file_name_in[:-2]
    os.rename(file_name, file_name_out + "_r" + file_extension)
    if "result" in output:
        dump_function(
            file_name=file_name_out + "_r" + file_extension,
            data_dict={"output": output["result"], "runtime": runtime},
        )
    else:
        dump_function(
            file_name=file_name_out + "_r" + file_extension,
            data_dict={"error": output["error"], "runtime": runtime},
        )
    os.rename(
        file_name_out + "_r" + file_extension, file_name_out + "_o" + file_extension
    )


def backend_execute_task_in_file(file_name: str) -> None:
    """
    Execute the task stored in a given HDF5 file.

    Args:
        file_name (str): The file name of the HDF5 file as an absolute path.

    Returns:
        None
    """
    file_extension = os.path.splitext(file_name)[1]
    if file_extension == ".h5":
        load_function = load_from_hdf
        dump_function = dump_to_hdf
    elif file_extension == ".json":
        from executorlib.standalone.json import dump_to_json, load_from_json

        load_function = load_from_json
        dump_function = dump_to_json
    else:
        raise ValueError("Unknown file extension!")
    apply_dict = backend_load_file(file_name=file_name, load_function=load_function)
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
        dump_function=dump_function,
    )


def _isinstance(obj: Any, cls: type) -> bool:
    return str(obj.__class__) == str(cls)
