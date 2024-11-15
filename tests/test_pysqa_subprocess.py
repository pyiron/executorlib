import unittest

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
