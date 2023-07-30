from abc import ABC
import os

from pympipool.shared.taskexecutor import cloudpickle_register, interface_bootup


class PoolBase(ABC):
    """
    Base class for the Pool and MPISpawnPool classes defined below. The PoolBase class is not intended to be used
    alone. Rather it implements the __enter__(), __exit__() and shutdown() function shared between the derived classes.
    """

    def __init__(self):
        self._future_dict = {}
        self._interface = None
        cloudpickle_register(ind=3)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False

    def shutdown(self, wait=True):
        self._interface.shutdown(wait=wait)


class Pool(PoolBase):
    """
    The pympipool.Pool behaves like the multiprocessing.Pool but it uses mpi4py to distribute tasks. In contrast to the
    mpi4py.futures.MPIPoolExecutor the pympipool.Pool can be executed in a serial python process and does not require
    the python script to be executed with MPI. Still internally the pympipool.Pool uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        max_workers (int): defines the total number of MPI ranks to use
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only)
        enable_flux_backend (bool): use the flux-framework as backend
        enable_slurm_backend (bool): enable the SLURM queueing system as backend - defaults to False
        cwd (str/None): current working directory where the parallel python task is executed
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

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
        max_workers=1,
        gpus_per_task=0,
        oversubscribe=False,
        enable_flux_backend=False,
        enable_slurm_backend=False,
        cwd=None,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        command_lst = [
            "python",
            "-m",
            "mpi4py.futures",
            os.path.abspath(
                os.path.join(__file__, "..", "..", "backend", "mpipool.py")
            ),
            "--cores-per-task",
            str(1),
            "--cores-total",
            str(max_workers),
        ]
        self._interface = interface_bootup(
            command_lst=command_lst,
            cwd=cwd,
            cores=max_workers,
            gpus_per_core=gpus_per_task,
            oversubscribe=oversubscribe,
            enable_flux_backend=enable_flux_backend,
            enable_slurm_backend=enable_slurm_backend,
            queue_adapter=queue_adapter,
            queue_adapter_kwargs=queue_adapter_kwargs,
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
            input_dict={
                "fn": func,
                "iterable": iterable,
                "chunksize": chunksize,
                "map": True,
            }
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
            input_dict={
                "fn": func,
                "iterable": iterable,
                "chunksize": chunksize,
                "map": False,
            }
        )


class MPISpawnPool(PoolBase):
    """
    The pympipool.MPISpawnPool behaves like the multiprocessing.Pool but it uses mpi4py to distribute tasks. In contrast
    to the mpi4py.futures.MPIPoolExecutor the pympipool.MPISpawnPool can be executed in a serial python process and does
    not require the python script to be executed with MPI. Still internally the pympipool.Pool uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        max_ranks (int): defines the total number of MPI ranks to use
        ranks_per_task (int): defines the number of MPI ranks per task
        gpus_per_task (int): number of GPUs per MPI rank - defaults to 0
        oversubscribe (bool): adds the `--oversubscribe` command line flag (OpenMPI only)
        cwd (str/None): current working directory where the parallel python task is executed
        queue_adapter (pysqa.queueadapter.QueueAdapter): generalized interface to various queuing systems
        queue_adapter_kwargs (dict/None): keyword arguments for the submit_job() function of the queue adapter

    Simple example:
        ```
        from pympipool import MPISpawnPool

        def calc(i, comm):
            return i, comm.Get_size(), comm.Get_rank()

        with MPISpawnPool(max_ranks=4, ranks_per_task=2) as p:
            print(p.map(func=calc, iterable=[1, 2, 3, 4]))
        ```
    """

    def __init__(
        self,
        max_ranks=1,
        ranks_per_task=1,
        gpus_per_task=0,
        oversubscribe=False,
        cwd=None,
        queue_adapter=None,
        queue_adapter_kwargs=None,
    ):
        super().__init__()
        executable = os.path.abspath(
            os.path.join(__file__, "..", "..", "backend", "mpipool.py")
        )
        if ranks_per_task == 1:
            command_lst = ["python", "-m", "mpi4py.futures", executable]
            cores = max_ranks
        else:
            # Running MPI parallel tasks within the map() requires mpi4py to use mpi spawn:
            # https://github.com/mpi4py/mpi4py/issues/324
            command_lst = ["python", executable]
            cores = 1
        command_lst += [
            "--cores-per-task",
            str(ranks_per_task),
            "--cores-total",
            str(max_ranks),
        ]
        self._interface = interface_bootup(
            command_lst=command_lst,
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_task,
            oversubscribe=oversubscribe,
            enable_flux_backend=False,
            enable_slurm_backend=False,
            queue_adapter=queue_adapter,
            queue_adapter_kwargs=queue_adapter_kwargs,
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
            input_dict={
                "fn": func,
                "iterable": iterable,
                "chunksize": chunksize,
                "map": True,
            }
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
            input_dict={
                "fn": func,
                "iterable": iterable,
                "chunksize": chunksize,
                "map": False,
            }
        )
