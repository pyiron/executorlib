import os

import cloudpickle

group_dict = {
    "fn": "function",
    "args": "input_args",
    "kwargs": "input_kwargs",
    "output": "output",
    "error": "error",
    "runtime": "runtime",
    "queue_id": "queue_id",
}


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


def get_cache_data(cache_directory: str) -> list[dict]:
    """
    Collect all HDF5 files in the cache directory

    Args:
        cache_directory (str): The directory to store cache files.

    Returns:
        list[dict]: List of dictionaries each representing on of the HDF5 files in the cache directory.
    """
    import h5py
    import numpy as np

    file_lst = []
    for file_name in get_cache_files(cache_directory=cache_directory):
        with h5py.File(file_name, "r") as hdf:
            file_content_dict = {
                key: cloudpickle.loads(np.void(hdf["/" + key]))
                for key in group_dict.values()
                if key in hdf
            }
        file_content_dict["filename"] = file_name
        file_lst.append(file_content_dict)
    return file_lst
