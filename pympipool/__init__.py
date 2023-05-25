import subprocess
import os
import socket
import inspect
import cloudpickle
import zmq
from concurrent.futures import Executor


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
        self, cores=1, cores_per_task=1, oversubscribe=False, enable_flux_backend=False
    ):
        self._cores = cores
        self._cores_per_task = cores_per_task
        self._process = None
        self._socket = None
        self._context = None
        self._enable_flux_backend = enable_flux_backend
        self._oversubscribe = oversubscribe
        self._bootup()

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """
        Map a given function on a list of attributes.

        Args:
            function: function to be applied to each element of the following list
            lst (list): list of arguments the function should be applied on

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # Cloudpickle can either pickle by value or pickle by reference. The functions which are communicated have to
        # be pickled by value rather than by reference, so the module which calls the map function is pickled by value.
        # https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
        # inspect can help to find the module which is calling pympipool
        # https://docs.python.org/3/library/inspect.html
        # to learn more about inspect another good read is:
        # http://pymotw.com/2/inspect/index.html#module-inspect
        # 1 refers to 1 level higher than the map function
        try:  # When executed in a jupyter notebook this can cause a ValueError - in this case we just ignore it.
            cloudpickle.register_pickle_by_value(
                inspect.getmodule(inspect.stack()[1][0])
            )
        except ValueError:
            pass
        self._send_raw(input_dict={"f": fn, "l": zip(*iterables)})
        return self._receive()

    def shutdown(self, wait=True, *, cancel_futures=False):
        if self._process is not None and self._process.poll() is None:
            self._send_raw(input_dict={"c": "close"})
            self._process.terminate()
            self._process.stdout.close()
            self._process.stdin.close()
            if wait:
                self._process.wait()
                self._socket.close()
                self._context.term()
                self._process = None
                self._socket = None
                self._context = None
        else:
            self._process = None
            self._socket = None
            self._context = None

    def _bootup(self):
        path = os.path.abspath(os.path.join(__file__, "..", "__main__.py"))
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PAIR)
        port_selected = self._socket.bind_to_random_port("tcp://*")
        if self._enable_flux_backend:
            command_lst = ["flux", "run"]
        else:
            command_lst = ["mpiexec"]
        if self._oversubscribe:
            command_lst += ["--oversubscribe"]
        if self._cores_per_task == 1:
            command_lst += ["-n", str(self._cores), "python", "-m", "mpi4py.futures"]
        else:
            command_lst += [
                "-n",
                "1",
                "python",
            ]
        command_lst += [path]
        if self._enable_flux_backend:
            command_lst += [
                "--host",
                socket.gethostname(),
            ]
        command_lst += [
            "--zmqport",
            str(port_selected),
            "--cores-per-task",
            str(self._cores_per_task),
            "--cores-total",
            str(self._cores),
        ]
        self._process = subprocess.Popen(
            command_lst,
            stdout=subprocess.PIPE,
            stderr=None,
            stdin=subprocess.PIPE,
        )

    def _send_raw(self, input_dict):
        self._socket.send(cloudpickle.dumps(input_dict))

    def _receive(self):
        output = cloudpickle.loads(self._socket.recv())
        if "r" in output.keys():
            return output["r"]
        else:
            error_type = output["et"].split("'")[1]
            raise eval(error_type)(output["e"])
