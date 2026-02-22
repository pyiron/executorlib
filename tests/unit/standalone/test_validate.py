import importlib
import inspect
import unittest
import os
import sys
from unittest.mock import patch

try:
    from pydantic import ValidationError
    skip_pydantic_test = False
except ImportError:
    skip_pydantic_test = True


class TestValidateFallback(unittest.TestCase):
    def test_validate_resource_dict_fallback(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            if 'executorlib.standalone.validate' in sys.modules:
                del sys.modules['executorlib.standalone.validate']

            from executorlib.standalone.validate import validate_resource_dict, ResourceDictValidation
            from dataclasses import is_dataclass

            self.assertTrue(is_dataclass(ResourceDictValidation))

            # Valid dict
            self.assertIsNone(validate_resource_dict({"cores": 1}))

            # Invalid dict (extra key)
            with self.assertRaises(TypeError):
                validate_resource_dict({"invalid_key": 1})

    def test_validate_resource_dict_with_optional_keys_fallback(self):
        with patch.dict('sys.modules', {'pydantic': None}):
            if 'executorlib.standalone.validate' in sys.modules:
                del sys.modules['executorlib.standalone.validate']

            from executorlib.standalone.validate import validate_resource_dict_with_optional_keys

            # Valid dict with optional keys
            with self.assertWarns(UserWarning):
                validate_resource_dict_with_optional_keys({"cores": 1, "optional_key": 2})

    def test_get_accepted_keys(self):
        from executorlib.standalone.validate import _get_accepted_keys, ResourceDictValidation

        accepted_keys = _get_accepted_keys(ResourceDictValidation)
        expected_keys = [
            "cores",
            "threads_per_core",
            "gpus_per_core",
            "cwd",
            "cache_key",
            "num_nodes",
            "exclusive",
            "error_log_file",
            "run_time_limit",
            "priority",
            "slurm_cmd_args"
        ]
        self.assertEqual(set(accepted_keys), set(expected_keys))
        with self.assertRaises(TypeError):
            _get_accepted_keys(int)


@unittest.skipIf(skip_pydantic_test, "pydantic is not installed")
class TestValidateFunction(unittest.TestCase):
    def test_validate_function(self):
        from executorlib import SingleNodeExecutor

        def dummy_function(i):
            return i
        
        with SingleNodeExecutor() as exe:
            with self.assertRaises(ValidationError):
                exe.submit(dummy_function, 5, resource_dict={"any": "thing"})
