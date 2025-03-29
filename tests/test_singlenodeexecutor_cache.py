import os
import shutil
import unittest

from executorlib import SingleNodeExecutor
from executorlib.standalone.serialize import cloudpickle_register

try:
    from executorlib import get_cache_data

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True


def get_error(a):
    raise ValueError(a)


@unittest.skipIf(
    skip_h5py_test, "h5py is not installed, so the h5io tests are skipped."
)
class TestCacheFunctions(unittest.TestCase):
    def test_cache_data(self):
        cache_directory = "./cache"
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            future_lst = [exe.submit(sum, [i, i]) for i in range(1, 4)]
            result_lst = [f.result() for f in future_lst]

        cache_lst = get_cache_data(cache_directory=cache_directory)
        self.assertEqual(sum([c["output"] for c in cache_lst]), sum(result_lst))
        self.assertEqual(
            sum([sum(c["input_args"][0]) for c in cache_lst]), sum(result_lst)
        )

    def test_cache_error(self):
        cache_directory = "./cache_error"
        with SingleNodeExecutor(cache_directory=cache_directory) as exe:
            cloudpickle_register(ind=1)
            f = exe.submit(get_error, a=1)
            with self.assertRaises(ValueError):
                print(f.result())

    def tearDown(self):
        if os.path.exists("cache"):
            shutil.rmtree("cache")
        if os.path.exists("cache_error"):
            shutil.rmtree("cache_error")
