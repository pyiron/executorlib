import subprocess
import os
import inspect
import cloudpickle


class Pool(object):
    """
    The pympipool.Pool behaves like the multiprocessing.Pool but it uses mpi4py to distribute tasks. In contrast to the
    mpi4py.futures.MPIPoolExecutor the pympipool.Pool can be executed in a serial python process and does not require
    the python script to be executed with MPI. Still internally the pympipool.Pool uses the
    mpi4py.futures.MPIPoolExecutor, consequently it is primarily an abstraction of its functionality to improve the
    usability in particular when used in combination with Jupyter notebooks.

    Args:
        cores (int): defines the number of MPI compute cores to use

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
    def __init__(self, cores=1):
        self._cores = cores
        self._process = None

    def __enter__(self):
        path = os.path.abspath(os.path.join(__file__, "..", "__main__.py"))
        self._process = subprocess.Popen(
            [
                "mpiexec",
                "--oversubscribe",
                "-n",
                str(self._cores),
                "python",
                "-m",
                "mpi4py.futures",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=None,
            stdin=subprocess.PIPE,
            # cwd=self.working_directory,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._send_raw(input_dict={"c": "close"})
        self._process.stdout.close()
        self._process.stdin.close()

    def map(self, function, lst):
        """
        Map a given function on a list of attributes.

        Args:
            function: function to be applied to each element of the following list
            lst (list): list of arguments the function should be applied on

        Returns:
            list: list of output generated from applying the function on the list of arguments
        """
        # Cloud pickle can decide which modules to pickle by value vs. pickle by reference
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
        self._send_raw(input_dict={"f": function, "l": lst})
        return self._receive()

    def _send_raw(self, input_dict):
        cloudpickle.dump(input_dict, self._process.stdin)
        self._process.stdin.flush()

    def _receive(self):
        output = cloudpickle.load(self._process.stdout)
        return output
