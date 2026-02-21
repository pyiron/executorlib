import importlib
import unittest
from unittest.mock import patch


class TestValidate(unittest.TestCase):
    def test_single_node_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            import executorlib.executor.single
            importlib.reload(executorlib.executor.single)

    def test_flux_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            import executorlib.executor.flux
            importlib.reload(executorlib.executor.flux)

    def test_slurm_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            import executorlib.executor.slurm
            importlib.reload(executorlib.executor.slurm)
