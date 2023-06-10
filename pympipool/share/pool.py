from concurrent.futures import Future

from pympipool.share.communication import SocketInterface
from pympipool.share.serial import get_parallel_subprocess_command, _cloudpickle_update


class PoolBase(object):
    def __init__(self):
        self._future_dict = {}
        self._interface = SocketInterface()
        _cloudpickle_update(ind=3)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._interface.shutdown(wait=wait)


class Pool(PoolBase):
    """
    The pympipool.Pool behaves like the multiprocessing.Pool but it uses mpi4py to distribute tasks. In contrast to the
    mpi4py.futures.MPIPoolExecutor the pympipool.Pool can be executed in a serial python process and does not require
    the python script to be executed with MPI. Still internally the pympipool.Pool uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the total number of MPI ranks to use
        cores_per_task (int): defines the number of MPI ranks per task
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only)

    Simple example:
        ```
        import numpy as np
        from pympipool import Pool

        def calc(i):
            return np.array(i ** 2)

        with Pool(cores=2) as p:
            print(p.map(func=calc, iterable=[1, 2, 3, 4]))
        ```
    """

    def __init__(
        self,
        cores=1,
        oversubscribe=False,
        enable_flux_backend=False,
        cwd=None,
    ):
        super().__init__()
        self._interface.bootup(
            command_lst=get_parallel_subprocess_command(
                port_selected=self._interface.bind_to_random_port(),
                cores=cores,
                cores_per_task=1,
                oversubscribe=oversubscribe,
                enable_flux_backend=enable_flux_backend,
                enable_mpi4py_backend=True,
            ),
            cwd=cwd,
        )

    def map(self, func, iterable, chunksize=None):
        """
        Map a given function on a list of attributes.

        Args:
            func: function to be applied to each element of the following list
            iterable (list): list of arguments the function should be applied on
            chunksize (int/None):

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # multiprocessing.pool.Pool and mpi4py.future.ExecutorPool have different defaults
        if chunksize is None:
            chunksize = 1
        return self._interface.send_and_receive_dict(
            input_dict={"fn": func, "iterable": iterable, "chunksize": chunksize, "map": True}
        )

    def starmap(self, func, iterable, chunksize=None):
        """
        Map a given function on a list of attributes.

        Args:
            func: function to be applied to each element of the following list
            iterable (list): list of arguments the function should be applied on
            chunksize (int/None):

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # multiprocessing.pool.Pool and mpi4py.future.ExecutorPool have different defaults
        if chunksize is None:
            chunksize = 1
        return self._interface.send_and_receive_dict(
            input_dict={"fn": func, "iterable": iterable, "chunksize": chunksize, "map": False}
        )


class PoolExtended(PoolBase):
    """
    The pympipool.Pool behaves like the multiprocessing.Pool but it uses mpi4py to distribute tasks. In contrast to the
    mpi4py.futures.MPIPoolExecutor the pympipool.Pool can be executed in a serial python process and does not require
    the python script to be executed with MPI. Still internally the pympipool.Pool uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the total number of MPI ranks to use
        cores_per_task (int): defines the number of MPI ranks per task
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only)

    Simple example:
        ```
        import numpy as np
        from pympipool import Pool

        def calc(i):
            return np.array(i ** 2)

        with Pool(cores=2) as p:
            print(p.map(func=calc, iterable=[1, 2, 3, 4]))
        ```
    """

    def __init__(
        self,
        cores=1,
        cores_per_task=1,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_mpi4py_backend=True,
        cwd=None,
    ):
        super().__init__()
        self._interface.bootup(
            command_lst=get_parallel_subprocess_command(
                port_selected=self._interface.bind_to_random_port(),
                cores=cores,
                cores_per_task=cores_per_task,
                oversubscribe=oversubscribe,
                enable_flux_backend=enable_flux_backend,
                enable_mpi4py_backend=enable_mpi4py_backend,
            ),
            cwd=cwd,
        )

    def map(self, func, iterable, chunksize=None):
        """
        Map a given function on a list of attributes.

        Args:
            func: function to be applied to each element of the following list
            iterable (list): list of arguments the function should be applied on
            chunksize (int/None):

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # multiprocessing.pool.Pool and mpi4py.future.ExecutorPool have different defaults
        if chunksize is None:
            chunksize = 1
        return self._interface.send_and_receive_dict(
            input_dict={"fn": func, "iterable": iterable, "chunksize": chunksize, "map": True}
        )

    def starmap(self, func, iterable, chunksize=None):
        """
        Map a given function on a list of attributes.

        Args:
            func: function to be applied to each element of the following list
            iterable (list): list of arguments the function should be applied on
            chunksize (int/None):

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # multiprocessing.pool.Pool and mpi4py.future.ExecutorPool have different defaults
        if chunksize is None:
            chunksize = 1
        return self._interface.send_and_receive_dict(
            input_dict={"fn": func, "iterable": iterable, "chunksize": chunksize, "map": False}
        )

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future_hash = self._interface.send_and_receive_dict(
            input_dict={"fn": fn, "args": args, "kwargs": kwargs}
        )
        self._future_dict[future_hash] = future
        return future

    def apply(self, fn, *args, **kwargs):
        return self._interface.send_and_receive_dict(
            input_dict={"fn": fn, "args": args, "kwargs": kwargs}
        )

    def update(self):
        hash_to_update = [h for h, f in self._future_dict.items() if not f.done()]
        if len(hash_to_update) > 0:
            self._interface.send_dict(input_dict={"update": hash_to_update})
            for k, v in self._interface.receive_dict().items():
                self._future_dict[k].set_result(v)
