import unittest
from executorlib.interactive.slurmspawner import generate_slurm_command

try:
    from executorlib.cache.queue_spawner import _pysqa_execute_command

    skip_pysqa_test = False
except ImportError:
    skip_pysqa_test = True


@unittest.skipIf(
    skip_pysqa_test, "pysqa is not installed, so the pysqa tests are skipped."
)
class TestPysqaExecuteCommand(unittest.TestCase):
    def test_pysqa_execute_command_list(self):
        out = _pysqa_execute_command(
            commands=["echo", "test"],
            working_directory=None,
            split_output=True,
            shell=True,
            error_filename="pysqa.err",
        )
        self.assertEqual(len(out), 2)
        self.assertEqual("test", out[0])

    def test_pysqa_execute_command_string(self):
        out = _pysqa_execute_command(
            commands="echo test",
            working_directory=None,
            split_output=False,
            shell=False,
            error_filename="pysqa.err",
        )
        self.assertEqual(len(out), 5)
        self.assertEqual("test\n", out)

    def test_pysqa_execute_command_fail(self):
        with self.assertRaises(FileNotFoundError):
            _pysqa_execute_command(
                commands=["no/executable/available"],
                working_directory=None,
                split_output=True,
                shell=False,
                error_filename="pysqa.err",
            )

    def test_generate_slurm_command(self):
        command_lst = generate_slurm_command(
            cores=1,
            cwd="/tmp/test",
            threads_per_core=2,
            gpus_per_core=1,
            num_nodes=1,
            exclusive=True,
            openmpi_oversubscribe=True,
            slurm_cmd_args=["--help"],
        )
        self.assertEqual(len(command_lst), 12)
        reply_lst = ['srun', '-n', '1', '-D', '/tmp/test', '-N', '1', '--cpus-per-task=2', '--gpus-per-task=1', '--exact', '--oversubscribe', '--help']
        self.assertEqual(command_lst, reply_lst)
