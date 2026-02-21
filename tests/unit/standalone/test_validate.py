import importlib
import inspect
import unittest
import os
import sys
from unittest.mock import patch


class TestValidate(unittest.TestCase):
    def test_single_node_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            if 'executorlib.standalone.validate' in sys.modules:
                del sys.modules['executorlib.standalone.validate']
            if 'executorlib.executor.single' in sys.modules:
                del sys.modules['executorlib.executor.single']

            import executorlib.executor.single
            importlib.reload(executorlib.executor.single)

            from executorlib.executor.single import validate_resource_dict
            
            source_file = inspect.getfile(validate_resource_dict)
            if os.name == 'nt':
                self.assertTrue(source_file.endswith('task_scheduler\\base.py'))
            else:
                self.assertTrue(source_file.endswith('task_scheduler/base.py'))
            self.assertIsNone(validate_resource_dict({"any": "thing"}))

    def test_flux_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            if 'executorlib.standalone.validate' in sys.modules:
                del sys.modules['executorlib.standalone.validate']
            if 'executorlib.executor.flux' in sys.modules:
                del sys.modules['executorlib.executor.flux']

            import executorlib.executor.flux
            importlib.reload(executorlib.executor.flux)

            from executorlib.executor.flux import validate_resource_dict
            
            source_file = inspect.getfile(validate_resource_dict)
            if os.name == 'nt':
                self.assertTrue(source_file.endswith('task_scheduler\\base.py'))
            else:
                self.assertTrue(source_file.endswith('task_scheduler/base.py'))
            self.assertIsNone(validate_resource_dict({"any": "thing"}))

    def test_slurm_job_executor(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            if 'executorlib.standalone.validate' in sys.modules:
                del sys.modules['executorlib.standalone.validate']
            if 'executorlib.executor.slurm' in sys.modules:
                del sys.modules['executorlib.executor.slurm']

            import executorlib.executor.slurm
            importlib.reload(executorlib.executor.slurm)

            from executorlib.executor.slurm import validate_resource_dict
            
            source_file = inspect.getfile(validate_resource_dict)
            if os.name == 'nt':
                self.assertTrue(source_file.endswith('task_scheduler\\base.py'))
            else:
                self.assertTrue(source_file.endswith('task_scheduler/base.py'))
            self.assertIsNone(validate_resource_dict({"any": "thing"}))
