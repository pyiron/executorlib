import os
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
    with h5py.File(file_name, "r") as hdf:
        if "output" in hdf:
            return True, True, cloudpickle.loads(np.void(hdf["/output"]))
        elif "error" in hdf:
            return True, False, cloudpickle.loads(np.void(hdf["/error"]))
        else:
            return False, False, None


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
