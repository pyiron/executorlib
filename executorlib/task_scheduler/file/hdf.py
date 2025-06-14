import os
from typing import Any, Optional

import cloudpickle
import h5py
import numpy as np

from executorlib.standalone.cache import group_dict


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
    if file_name is not None:
        with h5py.File(file_name, "r") as hdf:
            if "queue_id" in hdf:
                return cloudpickle.loads(np.void(hdf["/queue_id"]))
    return None
