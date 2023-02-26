import subprocess
import os
import inspect
import cloudpickle


class Pool(object):
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
        # Cloud pickle can decide which modules to pickle by value vs. pickle by reference
        # https://github.com/cloudpipe/cloudpickle#overriding-pickles-serialization-mechanism-for-importable-constructs
        # inspect can help to find the module which is calling pympipool
        # https://docs.python.org/3/library/inspect.html
        # to learn more about inspect another good read is:
        # http://pymotw.com/2/inspect/index.html#module-inspect
        # 1 refers to 1 level higher than the map function
        cloudpickle.register_pickle_by_value(inspect.getmodule(inspect.stack()[1][0]))
        self._send_raw(input_dict={"f": function, "l": lst})
        return self._receive()

    def _send_raw(self, input_dict):
        cloudpickle.dump(input_dict, self._process.stdin)
        self._process.stdin.flush()

    def _receive(self):
        output = cloudpickle.load(self._process.stdout)
        return output
