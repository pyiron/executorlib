import os
import unittest

from executorlib import Executor
from executorlib.shared.executor import cloudpickle_register
from executorlib.shell.executor import SubprocessExecutor

try:
    from conda.base.context import context

    skip_conda_test = False
except ImportError:
    skip_conda_test = True


def get_conda_env_prefix():
    return os.environ["CONDA_PREFIX"]


@unittest.skipIf(
    skip_conda_test, "conda is not installed, so the conda tests are skipped."
)
class CondaExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_name = "py312"
        if "envs" in context.root_prefix:
            cls.env_path = os.path.abspath(
                os.path.join(context.root_prefix, "..", cls.env_name)
            )
        else:
            cls.env_path = os.path.abspath(
                os.path.join(context.root_prefix, "envs", cls.env_name)
            )

    def test_shell_executor_conda(self):
        with SubprocessExecutor(
            max_workers=1, conda_environment_path=self.env_path
        ) as exe:
            future = exe.submit(["python", "--version"], universal_newlines=True)
            self.assertFalse(future.done())
            self.assertEqual("Python 3.12.1\n", future.result())
            self.assertTrue(future.done())

    def test_shell_executor_conda_name(self):
        with SubprocessExecutor(
            max_workers=1, conda_environment_name=self.env_name
        ) as exe:
            future = exe.submit(["python", "--version"], universal_newlines=True)
            self.assertFalse(future.done())
            self.assertEqual("Python 3.12.1\n", future.result())
            self.assertTrue(future.done())

    def test_python_executor_conda_path(self):
        with Executor(
            max_cores=1,
            hostname_localhost=True,
            backend="local",
            conda_environment_path=self.env_path,
        ) as exe:
            cloudpickle_register(ind=1)
            fs = exe.submit(get_conda_env_prefix)
            self.assertEqual(fs.result(), self.env_path)
            self.assertTrue(fs.done())

    def test_python_executor_conda_name(self):
        with Executor(
            max_cores=1,
            hostname_localhost=True,
            backend="local",
            conda_environment_name=self.env_name,
        ) as exe:
            cloudpickle_register(ind=1)
            fs = exe.submit(get_conda_env_prefix)
            self.assertEqual(fs.result(), self.env_path)
            self.assertTrue(fs.done())
