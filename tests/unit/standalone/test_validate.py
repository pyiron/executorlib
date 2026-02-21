import unittest
import warnings
from executorlib.executor.single import (
    validate_resource_dict,
    validate_resource_dict_with_optional_keys
)

try:
    from pydantic import ValidationError
    pydantic_installed = True
except ImportError:
    pydantic_installed = False

class TestValidate(unittest.TestCase):
    def test_validate_resource_dict(self):
        self.assertIsNone(validate_resource_dict(resource_dict={}))
        self.assertIsNone(validate_resource_dict(resource_dict={"cores": 1}))
        if pydantic_installed:
            with self.assertRaises(ValidationError):
                validate_resource_dict(resource_dict={"cores": "a"})

    def test_validate_resource_dict_with_optional_keys(self):
        # Test valid keys (should not warn)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_resource_dict_with_optional_keys(resource_dict={"cores": 1})
            self.assertEqual(len([item for item in w if issubclass(item.category, UserWarning)]), 0)

        # Test unknown keys
        if pydantic_installed:
            with self.assertWarns(UserWarning):
                validate_resource_dict_with_optional_keys(resource_dict={"unknown_key": 1})
            with self.assertRaises(ValidationError):
                # Valid key but wrong type
                validate_resource_dict_with_optional_keys(resource_dict={"cores": "a"})
        else:
            # Should not warn even with unknown keys if pydantic is missing (current fallback behavior)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                validate_resource_dict_with_optional_keys(resource_dict={"unknown_key": 1})
                self.assertEqual(len([item for item in w if issubclass(item.category, UserWarning)]), 0)
