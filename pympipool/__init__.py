import subprocess
import os
import dill
import inspect


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
        self._send(function=function, lst=lst)
        output = self._receive()
        return output

    def _send(self, function, lst):
        self._send_raw(input_dict={"f": inspect.getsource(function), "l": lst})

    def _send_raw(self, input_dict):
        dill.dump(input_dict, self._process.stdin)
        self._process.stdin.flush()

    def _receive(self):
        output = dill.load(self._process.stdout)
        return output
