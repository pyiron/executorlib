import unittest

from executorlib import get_cache_data
from executorlib.api import TestClusterExecutor
from executorlib.standalone.serialize import cloudpickle_register

try:
    import h5py

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def foo(x):
    return x + 1


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5io tests are skipped."
)
class TestTestClusterExecutor(unittest.TestCase):
    def test_cache_dir(self):
        with TestClusterExecutor(cache_directory="not_this_dir", resource_dict={}) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                foo,
                1,
                resource_dict={
                    "cache_directory": "rather_this_dir",
                    "cache_key": "foo",
                },
            )
            self.assertEqual(future.result(), 2)
        cache_lst = get_cache_data(cache_directory="not_this_dir")
        self.assertEqual(len(cache_lst), 0)
        cache_lst = get_cache_data(cache_directory="rather_this_dir")
        self.assertEqual(len(cache_lst), 1)
        with TestClusterExecutor(cache_directory="not_this_dir", resource_dict={}) as exe:
            cloudpickle_register(ind=1)
            future = exe.submit(
                foo,
                1,
                resource_dict={
                    "cache_directory": "rather_this_dir",
                    "cache_key": "foo",
                },
            )
            self.assertEqual(future.result(), 2)
        cache_lst = get_cache_data(cache_directory="not_this_dir")
        self.assertEqual(len(cache_lst), 0)
        cache_lst = get_cache_data(cache_directory="rather_this_dir")
        self.assertEqual(len(cache_lst), 1)
