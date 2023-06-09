from concurrent.futures import Executor, Future

from pympipool.share.communication import SocketInterface
from pympipool.share.serial import get_parallel_subprocess_command, _cloudpickle_update


class Pool(Executor):
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
            print(p.map(function=calc, lst=[1, 2, 3, 4]))
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
        self._future_dict = {}
        self._interface = SocketInterface()
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
        _cloudpickle_update(ind=2)

    def map(self, fn, iterables, timeout=None, chunksize=1):
        """
        Map a given function on a list of attributes.

        Args:
            fn: function to be applied to each element of the following list
            iterables (list): list of arguments the function should be applied on

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        return self._interface.send_and_receive_dict(
            input_dict={"f": fn, "l": iterables}
        )

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._interface.shutdown(wait=wait)

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future_hash = self._interface.send_and_receive_dict(
            input_dict={"f": fn, "a": args, "k": kwargs}
        )
        self._future_dict[future_hash] = future
        return future

    def apply(self, fn, *args, **kwargs):
        return self._interface.send_and_receive_dict(
            input_dict={"f": fn, "a": args, "k": kwargs}
        )

    def update(self):
        hash_to_update = [h for h, f in self._future_dict.items() if not f.done()]
        if len(hash_to_update) > 0:
            self._interface.send_dict(input_dict={"u": hash_to_update})
            for k, v in self._interface.receive_dict().items():
                self._future_dict[k].set_result(v)
