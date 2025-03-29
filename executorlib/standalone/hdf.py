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
}


def dump(file_name: Optional[str], data_dict: dict) -> None:
    """
    Dump data dictionary into HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path
        data_dict (dict): dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    if file_name is not None:
        with h5py.File(file_name, "a") as fname:
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
    if file_name is not None:
        with h5py.File(file_name, "r") as hdf:
            if "queue_id" in hdf:
                return cloudpickle.loads(np.void(hdf["/queue_id"]))
    return None


def get_cache_data(cache_directory: str) -> list[dict]:
    file_lst = []
    for task_key in os.listdir(cache_directory):
        file_name = os.path.join(cache_directory, task_key, "cache.h5out")
        os.makedirs(os.path.join(cache_directory, task_key), exist_ok=True)
        if os.path.exists(file_name):
            with h5py.File(file_name, "r") as hdf:
                file_content_dict = {
                    key: cloudpickle.loads(np.void(hdf["/" + key]))
                    for key in group_dict.values()
                    if key in hdf
                }
            file_content_dict["filename"] = file_name
            file_lst.append(file_content_dict)
    return file_lst
