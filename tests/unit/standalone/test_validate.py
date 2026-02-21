import unittest
from unittest.mock import patch


class TestValidate(unittest.TestCase):
    def test_single_node_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            from executorlib import SingleNodeExecutor

    def test_flux_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            from executorlib import FluxJobExecutor

    def test_slurm_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            from executorlib import SlurmJobExecutor
