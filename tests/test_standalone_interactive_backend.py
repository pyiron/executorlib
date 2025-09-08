import os
import sys
import unittest

from executorlib.standalone.interactive.backend import parse_arguments
from executorlib.standalone.interactive.spawner import MpiExecSpawner
from executorlib.task_scheduler.interactive.spawner_slurm import SrunSpawner

try:
    from executorlib.task_scheduler.interactive.spawner_pysqa import PysqaSpawner

    skip_pysqa_test = False
except ImportError:
    skip_pysqa_test = True


class TestParser(unittest.TestCase):
    def test_command_local(self):
        result_dict = {
            "host": "localhost",
            "worker_id": 0,
            "zmqport": "22",
        }
        command_lst = [
            "mpiexec",
            "-n",
            "2",
            "--oversubscribe",
            sys.executable,
            "/",
            "--zmqport",
            result_dict["zmqport"],
        ]
        interface = MpiExecSpawner(cwd=None, cores=2, openmpi_oversubscribe=True)
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[sys.executable, "/", "--zmqport", result_dict["zmqport"]]
            ),
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_slurm(self):
        result_dict = {
            "host": "127.0.0.1",
            "worker_id": 0,
            "zmqport": "22",
        }
        command_lst = [
            "srun",
            "-n",
            "2",
            "-D",
            os.path.abspath("."),
            "--gpus-per-task=1",
            "--oversubscribe",
            sys.executable,
            "/",
            "--host",
            result_dict["host"],
            "--zmqport",
            result_dict["zmqport"],
        ]
        interface = SrunSpawner(
            cwd=os.path.abspath("."),
            cores=2,
            gpus_per_core=1,
            openmpi_oversubscribe=True,
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[
                    sys.executable,
                    "/",
                    "--host",
                    result_dict["host"],
                    "--zmqport",
                    result_dict["zmqport"],
                ]
            ),
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    def test_command_slurm_user_command(self):
        result_dict = {
            "host": "127.0.0.1",
            "worker_id": 0,
            "zmqport": "22",
        }
        command_lst = [
            "srun",
            "-n",
            "2",
            "-D",
            os.path.abspath("."),
            "--mpi=pmi2",
            "--gpus-per-task=1",
            "--oversubscribe",
            "--account=test",
            "--job-name=executorlib",
            sys.executable,
            "/",
            "--host",
            result_dict["host"],
            "--zmqport",
            result_dict["zmqport"],
        ]
        interface = SrunSpawner(
            cwd=os.path.abspath("."),
            cores=2,
            gpus_per_core=1,
            openmpi_oversubscribe=True,
            slurm_cmd_args=["--account=test", "--job-name=executorlib"],
            pmi_mode="pmi2",
        )
        self.assertEqual(
            command_lst,
            interface.generate_command(
                command_lst=[
                    sys.executable,
                    "/",
                    "--host",
                    result_dict["host"],
                    "--zmqport",
                    result_dict["zmqport"],
                ]
            ),
        )
        self.assertEqual(result_dict, parse_arguments(command_lst))

    @unittest.skipIf(skip_pysqa_test, "pysqa is not installed, so the pysqa tests are skipped.")
    def test_command_pysqa(self):
        interface_slurm = PysqaSpawner(backend="slurm", cores=2, pmi_mode="pmix", num_nodes=2, threads_per_core=2, gpus_per_core=1, exclusive=True, openmpi_oversubscribe=True, slurm_cmd_args=["test"])
        output = ['srun', '-n', '2', '--mpi=pmix', '-N', '2', '--cpus-per-task=2', '--gpus-per-task=1', '--exact', '--oversubscribe', 'test']
        self.assertEqual(interface_slurm.generate_command(command_lst=[]), output)

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix")
        output = ['flux', 'run', '-n', '2', '-o', 'pmi=pmix']
        self.assertEqual(interface_flux.generate_command(command_lst=[]), output)

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix", num_nodes=2)
        with self.assertRaises(ValueError):
            interface_flux.generate_command(command_lst=[])

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix", threads_per_core=2)
        with self.assertRaises(ValueError):
            interface_flux.generate_command(command_lst=[])

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix", gpus_per_core=1)
        with self.assertRaises(ValueError):
            interface_flux.generate_command(command_lst=[])

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix", exclusive=True)
        with self.assertRaises(ValueError):
            interface_flux.generate_command(command_lst=[])

        interface_flux = PysqaSpawner(backend="flux", cores=2, pmi_mode="pmix", openmpi_oversubscribe=True)
        with self.assertRaises(ValueError):
            interface_flux.generate_command(command_lst=[])

        interface_nobackend = PysqaSpawner(cores=2)
        with self.assertRaises(ValueError):
            interface_nobackend.generate_command(command_lst=[])
        