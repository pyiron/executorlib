import os
import importlib
import unittest
import shutil
from time import sleep

from executorlib import SlurmClusterExecutor
from executorlib.standalone.serialize import cloudpickle_register

if shutil.which("srun") is not None:
    skip_slurm_test = False
else:
    skip_slurm_test = True

skip_mpi4py_test = importlib.util.find_spec("mpi4py") is None

try:
    from executorlib.standalone.hdf import dump

    skip_h5py_test = False
except ImportError:
    skip_h5py_test = True

try:
    import pysqa

    skip_pysqa_test = False
except ImportError:
    skip_pysqa_test = True

submission_template = """\
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env=L
#SBATCH --ntasks={{cores}}
{%- if dependency_list %}
#SBATCH --dependency=afterok:{{ dependency_list | join(',') }}
{%- endif %}

{{command}}
"""


def mpi_funct(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


def add_with_sleep(parameter_1, parameter_2):
    sleep(1)
    return parameter_1 + parameter_2


@unittest.skipIf(
    skip_slurm_test or skip_mpi4py_test or skip_h5py_test,
    "h5py or mpi4py or SLRUM are not installed, so the h5py, slurm and mpi4py tests are skipped.",
)
class TestCacheExecutorPysqa(unittest.TestCase):
    def test_executor(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache", "submission_template": submission_template},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode="pmi2",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 3)
            self.assertTrue(fs1.done())

    def test_executor_dependencies(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 1, "cwd": "executorlib_cache", "submission_template": submission_template},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode="pmi2",
        ) as exe:
            fs1 = exe.submit(add_with_sleep, 1, 1)
            fs2 = exe.submit(add_with_sleep, fs1, 1)
            fs3 = exe.submit(add_with_sleep, fs1, fs2)
            self.assertFalse(fs1.done())
            self.assertFalse(fs2.done())
            self.assertFalse(fs3.done())
            self.assertEqual(fs1.result(), 2)
            self.assertEqual(fs2.result(), 3)
            self.assertEqual(fs3.result(), 5)
            self.assertEqual(len(os.listdir("executorlib_cache")), 6)
            self.assertTrue(fs1.done())
            self.assertTrue(fs2.done())
            self.assertTrue(fs3.done())

    def test_executor_no_cwd(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "submission_template": submission_template},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode="pmi2",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertEqual(len(os.listdir("executorlib_cache")), 2)
            self.assertTrue(fs1.done())

    def test_executor_existing_files(self):
        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache", "submission_template": submission_template},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode="pmi2",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())
            self.assertEqual(len(os.listdir("executorlib_cache")), 3)
            for file_name in os.listdir("executorlib_cache"):
                file_path = os.path.join("executorlib_cache", file_name )
                os.remove(file_path)
                if ".h5" in file_path:
                    task_key = file_path[:-5] + "_i.h5"
                    dump(file_name=task_key, data_dict={"a": 1})

        with SlurmClusterExecutor(
            resource_dict={"cores": 2, "cwd": "executorlib_cache", "submission_template": submission_template},
            block_allocation=False,
            cache_directory="executorlib_cache",
            pmi_mode="pmi2",
        ) as exe:
            cloudpickle_register(ind=1)
            fs1 = exe.submit(mpi_funct, 1)
            self.assertFalse(fs1.done())
            self.assertEqual(fs1.result(), [(1, 2, 0), (1, 2, 1)])
            self.assertTrue(fs1.done())
            self.assertEqual(len(os.listdir("executorlib_cache")), 3)

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)


@unittest.skipIf(skip_pysqa_test, "pysqa is not installed, so the pysqa tests are skipped.")
class TestSlurmClusterInit(unittest.TestCase):
    def test_slurm_cluster_block_allocation(self):
        with self.assertRaises(ValueError):
            SlurmClusterExecutor(block_allocation=True)

    def test_slurm_cluster_file(self):
        self.assertTrue(SlurmClusterExecutor(block_allocation=False))