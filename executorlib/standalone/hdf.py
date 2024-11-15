from typing import Optional, Tuple

import cloudpickle
import h5py
import numpy as np


def dump(file_name: str, data_dict: dict) -> None:
    """
    Dump data dictionary into HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path
        data_dict (dict): dictionary containing the python function to be executed {"fn": ..., "args": (), "kwargs": {}}
    """
    group_dict = {
        "fn": "function",
        "args": "input_args",
        "kwargs": "input_kwargs",
        "output": "output",
        "queue_id": "queue_id",
    }
    with h5py.File(file_name, "a") as fname:
        for data_key, data_value in data_dict.items():
            if data_key in group_dict.keys():
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


def get_output(file_name: str) -> Tuple[bool, object]:
    """
    Check if output is available in the HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path

    Returns:
        Tuple[bool, object]: boolean flag indicating if output is available and the output object itself
    """
    with h5py.File(file_name, "r") as hdf:
        if "output" in hdf:
            return True, cloudpickle.loads(np.void(hdf["/output"]))
        else:
            return False, None


def get_queue_id(file_name: str) -> Optional[int]:
    with h5py.File(file_name, "r") as hdf:
        if "queue_id" in hdf:
            return cloudpickle.loads(np.void(hdf["/queue_id"]))
        else:
            return None
