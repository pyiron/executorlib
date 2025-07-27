import os
import importlib.util
import sys
import unittest
from executorlib.standalone.command import get_cache_execute_command, get_interactive_execute_command


skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None


class TestCommands(unittest.TestCase):
    def test_get_interactive_execute_command_serial(self):
        output = get_interactive_execute_command(cores=1)
        self.assertEqual(output[0], sys.executable)
        self.assertEqual(output[1].split(os.sep)[-1], "interactive_serial.py")

    @unittest.skipIf(
        skip_mpi4py_test,
        "mpi4py is not installed, so the mpi4py tests are skipped.",
    )
    def test_get_interactive_execute_command_parallel(self):
        output = get_interactive_execute_command(cores=2)
        self.assertEqual(output[0], sys.executable)
        self.assertEqual(output[1].split(os.sep)[-1], "interactive_parallel.py")

    def test_get_cache_execute_command_serial(self):
        file_name = "test.txt"
        output = get_cache_execute_command(cores=1, file_name=file_name)
        self.assertEqual(output[0], sys.executable)
        self.assertEqual(output[1].split(os.sep)[-1], "cache_serial.py")
        self.assertEqual(output[2], file_name)
        output = get_cache_execute_command(cores=1, file_name=file_name, backend="slurm")
        self.assertEqual(output[0], sys.executable)
        self.assertEqual(output[1].split(os.sep)[-1], "cache_serial.py")
        self.assertEqual(output[2], file_name)
        output = get_cache_execute_command(cores=1, file_name=file_name, backend="flux")
        self.assertEqual(output[0], sys.executable)
        self.assertEqual(output[1].split(os.sep)[-1], "cache_serial.py")
        self.assertEqual(output[2], file_name)

    @unittest.skipIf(
        skip_mpi4py_test,
        "mpi4py is not installed, so the mpi4py tests are skipped.",
    )
    def test_get_cache_execute_command_parallel(self):
        file_name = "test.txt"
        output = get_cache_execute_command(cores=2, file_name=file_name)
        self.assertEqual(output[0], "mpiexec")
        self.assertEqual(output[1], "-n")
        self.assertEqual(output[2], str(2))
        self.assertEqual(output[3], sys.executable)
        self.assertEqual(output[4].split(os.sep)[-1], "cache_parallel.py")
        self.assertEqual(output[5], file_name)
        output = get_cache_execute_command(cores=2, file_name=file_name, backend="slurm")
        self.assertEqual(output[0], "srun")
        self.assertEqual(output[1], "-n")
        self.assertEqual(output[2], str(2))
        self.assertEqual(output[3], "--mpi=pmi2")
        self.assertEqual(output[4], sys.executable)
        self.assertEqual(output[5].split(os.sep)[-1], "cache_parallel.py")
        self.assertEqual(output[6], file_name)
        output = get_cache_execute_command(cores=2, file_name=file_name, backend="flux")
        self.assertEqual(output[0], "flux")
        self.assertEqual(output[1], "run")
        self.assertEqual(output[2], "-n")
        self.assertEqual(output[3], str(2))
        self.assertEqual(output[4], sys.executable)
        self.assertEqual(output[5].split(os.sep)[-1], "cache_parallel.py")
        self.assertEqual(output[6], file_name)
        output = get_cache_execute_command(cores=2, file_name=file_name, backend="flux", executor_pmi_mode="pmix")
        self.assertEqual(output[0], "flux")
        self.assertEqual(output[1], "run")
        self.assertEqual(output[2], "-o")
        self.assertEqual(output[3], "pmi=pmix")
        self.assertEqual(output[4], "-n")
        self.assertEqual(output[5], str(2))
        self.assertEqual(output[6], sys.executable)
        self.assertEqual(output[7].split(os.sep)[-1], "cache_parallel.py")
        self.assertEqual(output[8], file_name)
        with self.assertRaises(ValueError):
            get_cache_execute_command(cores=2, file_name=file_name, backend="test")
