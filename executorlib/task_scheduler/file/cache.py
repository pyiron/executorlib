import os

import cloudpickle
import numpy as np

from executorlib.standalone.hdf import group_dict


def get_cache_data(cache_directory: str) -> list[dict]:
    """
    Collect all HDF5 files in the cache directory

    Args:
        cache_directory (str): The directory to store cache files.

    Returns:
        list[dict]: List of dictionaries each representing on of the HDF5 files in the cache directory.
    """
    import h5py

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
