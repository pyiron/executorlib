import os
from concurrent.futures import Future
from time import sleep
from typing import Any, Optional

import cloudpickle
import h5py
import numpy as np

group_dict = {
    "fn": "function",
    "args": "input_args",
    "kwargs": "input_kwargs",
    "output": "output",
    "error": "error",
    "resource_dict": "resource_dict",
    "runtime": "runtime",
    "queue_id": "queue_id",
    "error_log_file": "error_log_file",
}


def dump(file_name: Optional[str], data_dict: dict) -> None:
    """
    Dump data dictionary into HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path
        data_dict (dict): dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    if file_name is not None:
        file_name_abs = os.path.abspath(file_name)
        os.makedirs(os.path.dirname(file_name_abs), exist_ok=True)
        with h5py.File(file_name_abs, "a") as fname:
            for data_key, data_value in data_dict.items():
                if data_key in group_dict:
                    fname.create_dataset(
                        name="/" + group_dict[data_key],
                        data=np.void(cloudpickle.dumps(data_value)),
                    )


def load(file_name: str) -> dict:
    """
    Load data dictionary from HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path

    Returns:
        dict: dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    with h5py.File(file_name, "r") as hdf:
        data_dict = {}
        if "function" in hdf:
            data_dict["fn"] = cloudpickle.loads(np.void(hdf["/function"]))
        else:
            raise TypeError("Function not found in HDF5 file.")
        if "input_args" in hdf:
            data_dict["args"] = cloudpickle.loads(np.void(hdf["/input_args"]))
        else:
            data_dict["args"] = ()
        if "input_kwargs" in hdf:
            data_dict["kwargs"] = cloudpickle.loads(np.void(hdf["/input_kwargs"]))
        else:
            data_dict["kwargs"] = {}
        if "resource_dict" in hdf:
            data_dict["resource_dict"] = cloudpickle.loads(
                np.void(hdf["/resource_dict"])
            )
        else:
            data_dict["resource_dict"] = {}
        if "error_log_file" in hdf:
            data_dict["error_log_file"] = cloudpickle.loads(
                np.void(hdf["/error_log_file"])
            )
        return data_dict


def get_output(file_name: str) -> tuple[bool, bool, Any]:
    """
    Check if output is available in the HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path

    Returns:
        Tuple[bool, bool, object]: boolean flag indicating if output is available and the output object itself
    """
    def get_output_helper(file_name: str) -> tuple[bool, bool, Any]:
        with h5py.File(file_name, "r") as hdf:
            if "output" in hdf:
                return True, True, cloudpickle.loads(np.void(hdf["/output"]))
            elif "error" in hdf:
                return True, False, cloudpickle.loads(np.void(hdf["/error"]))
            else:
                return False, False, None
            
    i = 0
    while i < 10:
        try:
            return get_output_helper(file_name=file_name)
        except FileNotFoundError as e:
            i += 1
            sleep(0.1)
    raise e

def get_runtime(file_name: str) -> float:
    """
    Get run time from HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path

    Returns:
        float: run time from the execution of the python function
    """
    with h5py.File(file_name, "r") as hdf:
        if "runtime" in hdf:
            return cloudpickle.loads(np.void(hdf["/runtime"]))
        else:
            return 0.0


def get_queue_id(file_name: Optional[str]) -> Optional[int]:
    """
    Get queuing system id from HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path

    Returns:
        int: queuing system id from the execution of the python function
    """
    if file_name is not None and os.path.exists(file_name):
        with h5py.File(file_name, "r") as hdf:
            if "queue_id" in hdf:
                return cloudpickle.loads(np.void(hdf["/queue_id"]))
    return None


def get_cache_data(cache_directory: str) -> list[dict]:
    """
    Collect all HDF5 files in the cache directory

    Args:
        cache_directory (str): The directory to store cache files.

    Returns:
        list[dict]: List of dictionaries each representing on of the HDF5 files in the cache directory.
    """
    return [
        _get_content_of_file(file_name=file_name) | {"filename": file_name}
        for file_name in get_cache_files(cache_directory=cache_directory)
    ]


def get_cache_files(cache_directory: str) -> list[str]:
    """
    Recursively find all HDF5 files in the cache_directory which contain outputs.

    Args:
        cache_directory (str): The directory to store cache files.

    Returns:
        list[str]: List of HDF5 file in the cache directory which contain outputs.
    """
    file_lst = []
    cache_directory_abs = os.path.abspath(cache_directory)
    for dirpath, _, filenames in os.walk(cache_directory_abs):
        file_lst += [os.path.join(dirpath, f) for f in filenames if f.endswith("_o.h5")]
    return file_lst


def get_future_from_cache(
    cache_directory: str,
    cache_key: str,
) -> Future:
    """
    Reload future from HDF5 file in cache directory with the given cache key. The function checks if the output file
    exists, if not it checks for the input file. If neither of them exist, it raises a FileNotFoundError. If the output
    file exists, it loads the output and sets it as the result of the future. If only the input file exists, it checks
    if the execution is finished and if there was an error. If there was no error, it sets the output as the result of
    the future, otherwise it raises the error.

    Args:
        cache_directory (str): The directory to store cache files.
        cache_key (str): The key of the cache file to be reloaded.

    Returns:
        Future: Future object containing the result of the execution of the python function.
    """
    file_name_in = os.path.join(cache_directory, cache_key + "_i.h5")
    file_name_out = os.path.join(cache_directory, cache_key + "_o.h5")
    future: Future = Future()
    if os.path.exists(file_name_out):
        file_name = file_name_out
    elif os.path.exists(file_name_in):
        file_name = file_name_in
    else:
        raise FileNotFoundError(
            f"Neither input nor output file for cache key {cache_key} found in cache directory {cache_directory}."
        )
    exec_flag, no_error_flag, result = get_output(file_name=file_name)
    if exec_flag and no_error_flag:
        future.set_result(result)
    elif exec_flag:
        raise result
    return future


def _get_content_of_file(file_name: str) -> dict:
    """
    Get content of an HDF5 file

    Args:
        file_name (str): file name

    Returns:
        dict: Content of HDF5 file
    """
    with h5py.File(file_name, "r") as hdf:
        return {
            key: cloudpickle.loads(np.void(hdf["/" + key]))
            for key in group_dict.values()
            if key in hdf
        }
