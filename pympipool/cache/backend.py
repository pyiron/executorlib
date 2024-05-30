import os

from pympipool.cache.hdf import dump, load
from pympipool.cache.shared import FutureItem


def execute_task_in_file(file_name: str):
    """
    Execute the task stored in a given HDF5 file

    Args:
        file_name (str): file name of the HDF5 file as absolute path
    """
    apply_dict = load(file_name=file_name)
    args = [
        arg if not isinstance(arg, FutureItem) else arg.result()
        for arg in apply_dict["args"]
    ]
    kwargs = {
        key: arg if not isinstance(arg, FutureItem) else arg.result()
        for key, arg in apply_dict["kwargs"].items()
    }
    result = apply_dict["fn"].__call__(*args, **kwargs)
    file_name_out = os.path.splitext(file_name)[0]
    os.rename(file_name, file_name_out + ".h5ready")
    dump(file_name=file_name_out + ".h5ready", data_dict={"output": result})
    os.rename(file_name_out + ".h5ready", file_name_out + ".h5out")
