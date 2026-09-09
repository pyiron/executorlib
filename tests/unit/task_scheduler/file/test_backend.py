from concurrent.futures import Future
import os
import shutil
import sys
import unittest
from unittest.mock import patch

from executorlib.standalone.select import FutureSelector


try:
    from executorlib.task_scheduler.file.backend import backend_execute_task_in_file, backend_write_file
    from executorlib.task_scheduler.file.shared import _check_task_output, _convert_args_and_kwargs, FutureItem
    from executorlib.standalone.hdf import dump, get_runtime
    from executorlib.standalone.serialize import serialize_funct

    skip_h5io_test = False
except ImportError:
    skip_h5io_test = True

try:
    from pandas import DataFrame
    import pysqa  # noqa: F401
    from executorlib.standalone.command_pysqa import pysqa_job_output_validation
    from executorlib.task_scheduler.file.spawner_pysqa import (
        _merge_cache_with_queue_status,
        get_queue_system_cache_data,
    )

    skip_pysqa_test = False
except ImportError:
    skip_pysqa_test = True


def my_funct(a, b):
    return a + b


def return_dict(a, b):
    return {"a": a, "b": b}


def return_list(a, b):
    return [a, b]


def get_error(a):
    raise ValueError(a)


@unittest.skipIf(
    skip_h5io_test, "h5io is not installed, so the h5io tests are skipped."
)
class TestSharedFunctions(unittest.TestCase):
    def test_execute_function_mixed(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory,
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_backend_write_file(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_file_i.h5")
        dump(file_name=file_name, data_dict={"fn": my_funct, "args": [1], "kwargs": {"b": 2}})
        backend_write_file(file_name=file_name, output={"result": 3}, runtime=0.1)
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, "test_file_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_backend_write_file_serialization_error(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        file_name = os.path.join(cache_directory, "test_file_i.h5")
        dump(file_name=file_name, data_dict={"fn": my_funct, "args": [1], "kwargs": {"b": 2}})
        backend_write_file(file_name=file_name, output={"result": Future()}, runtime=0.1)
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, "test_file_o.h5")
        )
        with self.assertRaises(Exception):
            future_file_obj.result()

    def test_execute_function_mixed_selector_convert(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key_1, data_dict = serialize_funct(
            fn=return_dict,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name_1 = os.path.join(cache_directory, task_key_1 + "_i.h5")
        dump(file_name=file_name_1, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name_1)
        f1 = Future()
        _check_task_output(
            task_key=task_key_1, future_obj=f1, cache_directory=cache_directory,
        )
        task_key_2, data_dict = serialize_funct(
            fn=return_list,
            fn_args=[1],
            fn_kwargs={"b": 2},
        )
        file_name_2 = os.path.join(cache_directory, task_key_2 + "_i.h5")
        dump(file_name=file_name_2, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name_2)
        f2 = Future()
        _check_task_output(
            task_key=task_key_2, future_obj=f2, cache_directory=cache_directory,
        )
        fs1 = FutureSelector(future=f1, selector="a")
        fs2 = FutureSelector(future=f2, selector=1)
        task_args, task_kwargs, future_wait_key_lst = _convert_args_and_kwargs(
            task_dict={"fn": 1, "args": (fs1,), "kwargs": {"b": fs2}},
            memory_dict={task_key_1: f1, task_key_2: f2},
            file_name_dict={
                task_key_1: os.path.join(cache_directory, task_key_1 + "_o.h5"), 
                task_key_2: os.path.join(cache_directory, task_key_2 + "_o.h5"),
            },
        )
        self.assertEqual(task_args[0].result(), 1)
        self.assertEqual(task_kwargs["b"].result(), 2)
        self.assertTrue(len(future_wait_key_lst) == 2)

    def test_execute_function_args(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=[1, 2],
            fn_kwargs=None,
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        os.makedirs(os.path.join(cache_directory, task_key), exist_ok=True)
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory,
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_kwargs(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=my_funct,
            fn_args=None,
            fn_kwargs={"a": 1, "b": 2},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory,
        )
        self.assertTrue(future_obj.done())
        self.assertEqual(future_obj.result(), 3)
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        self.assertEqual(future_file_obj.result(), 3)

    def test_execute_function_error(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(
            fn=get_error,
            fn_args=[],
            fn_kwargs={"a": 1},
        )
        file_name = os.path.join(cache_directory, task_key + "_i.h5")
        data_dict["error_log_file"] = os.path.join(cache_directory, "error.out")
        dump(file_name=file_name, data_dict=data_dict)
        backend_execute_task_in_file(file_name=file_name)
        future_obj = Future()
        _check_task_output(
            task_key=task_key, future_obj=future_obj, cache_directory=cache_directory,
        )
        self.assertTrue(future_obj.done())
        with self.assertRaises(ValueError):
            future_obj.result()
        with open(os.path.join(cache_directory, "error.out"), "r") as f:
            content = f.readlines()
        self.assertEqual(content[1], 'args: []\n')
        self.assertEqual(content[2], "kwargs: {'a': 1}\n")
        self.assertEqual(content[-1], 'ValueError: 1\n')
        self.assertTrue(
            get_runtime(file_name=os.path.join(cache_directory, task_key + "_o.h5"))
            > 0.0
        )
        future_file_obj = FutureItem(
            file_name=os.path.join(cache_directory, task_key + "_o.h5")
        )
        self.assertTrue(future_file_obj.done())
        with self.assertRaises(ValueError):
            future_file_obj.result()

    @unittest.skipIf(sys.platform == "win32" or skip_pysqa_test, "pysqa module patching not supported on Windows or when pysqa is not installed")
    def test_check_task_output_dead_job_without_output(self):
        # Reproduces https://github.com/pyiron/executorlib/issues/1037 : a queuing system job
        # which dies without ever writing its output file (e.g. walltime TIMEOUT, OOM, NODE_FAIL
        # or an external scancel) must fail the future instead of leaving it pending forever.
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(fn=my_funct, fn_args=[1], fn_kwargs={"b": 2})
        future_obj = Future()
        with patch(
            "executorlib.standalone.command_pysqa.pysqa_get_status_of_job",
            return_value=None,
        ) as status_mock:
            _check_task_output(
                task_key=task_key,
                future_obj=future_obj,
                cache_directory=cache_directory,
                queue_id=123,
                backend="slurm",
                validate_function=pysqa_job_output_validation,
            )
        status_mock.assert_called_once()
        self.assertTrue(future_obj.done())
        with self.assertRaises(RuntimeError):
            future_obj.result()

    @unittest.skipIf(sys.platform == "win32" or skip_pysqa_test, "pysqa module patching not supported on Windows or when pysqa is not installed")
    def test_check_task_output_dead_job_reported_as_error_status(self):
        # On some backends (e.g. flux, whose "flux jobs -a" keeps listing inactive jobs) a
        # cancelled/failed job is never removed from the queue listing - unlike slurm's squeue,
        # which drops the job entirely (status None) once it is gone. Instead pysqa reports it
        # with its terminal-failure status "error" (see command_pysqa.pysqa_terminate, which
        # already treats "error" as not alive). That status must also fail the future - see
        # https://github.com/pyiron/executorlib/issues/1037.
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(fn=my_funct, fn_args=[1], fn_kwargs={"b": 2})
        future_obj = Future()
        with patch(
            "executorlib.standalone.command_pysqa.pysqa_get_status_of_job",
            return_value="error",
        ) as status_mock:
            _check_task_output(
                task_key=task_key,
                future_obj=future_obj,
                cache_directory=cache_directory,
                queue_id=123,
                backend="flux",
                validate_function=pysqa_job_output_validation,
            )
        status_mock.assert_called_once()
        self.assertTrue(future_obj.done())
        with self.assertRaises(RuntimeError):
            future_obj.result()

    @unittest.skipIf(sys.platform == "win32" or skip_pysqa_test, "pysqa module patching not supported on Windows or when pysqa is not installed")
    def test_check_task_output_job_still_running(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(fn=my_funct, fn_args=[1], fn_kwargs={"b": 2})
        future_obj = Future()
        with patch(
            "executorlib.standalone.command_pysqa.pysqa_get_status_of_job",
            return_value="running",
        ) as status_mock:
            _check_task_output(
                task_key=task_key,
                future_obj=future_obj,
                cache_directory=cache_directory,
                queue_id=123,
                backend="slurm",
                validate_function=pysqa_job_output_validation,
            )
        status_mock.assert_called_once()
        self.assertFalse(future_obj.done())

    @unittest.skipIf(sys.platform == "win32" or skip_pysqa_test, "pysqa module patching not supported on Windows or when pysqa is not installed")
    def test_check_task_output_status_check_is_throttled(self):
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(fn=my_funct, fn_args=[1], fn_kwargs={"b": 2})
        status_check_dict = {}
        with patch(
            "executorlib.standalone.command_pysqa.pysqa_get_status_of_job",
            return_value="running",
        ) as status_mock:
            for _ in range(3):
                _check_task_output(
                    task_key=task_key,
                    future_obj=Future(),
                    cache_directory=cache_directory,
                    queue_id=123,
                    backend="slurm",
                    validate_function=pysqa_job_output_validation,
                    status_check_dict=status_check_dict,
                )
        status_mock.assert_called_once()

    @unittest.skipIf(sys.platform == "win32" or skip_pysqa_test, "pysqa module patching not supported on Windows or when pysqa is not installed")
    def test_check_task_output_no_backend_never_queries_status(self):
        # subprocess-backed tasks (backend=None) must never trigger a queuing system status check.
        cache_directory = os.path.abspath("executorlib_cache")
        os.makedirs(cache_directory, exist_ok=True)
        task_key, data_dict = serialize_funct(fn=my_funct, fn_args=[1], fn_kwargs={"b": 2})
        future_obj = Future()
        with patch(
            "executorlib.standalone.command_pysqa.pysqa_get_status_of_job",
        ) as status_mock:
            _check_task_output(
                task_key=task_key,
                future_obj=future_obj,
                cache_directory=cache_directory,
                queue_id=123,
                backend=None,
            )
        status_mock.assert_not_called()
        self.assertFalse(future_obj.done())

    @unittest.skipIf(
        sys.platform == "win32" or skip_pysqa_test,
        "pysqa module patching not supported on Windows or when pysqa is not installed",
    )
    def test_merge_cache_with_queue_status_edge_cases(self):
        cache_lst = _merge_cache_with_queue_status(
            cache_dict=[
                {"filename": "finished_o.h5", "output": 1},
                {"filename": "queued_i.h5", "queue_id": 42},
                {"filename": "local_i.h5"},
                {"filename": "aborted_i.h5", "queue_id": 99},
            ],
            dataframe=DataFrame(
                [
                    {"jobid": 42, "status": "submitted"},
                    {"jobid": 42, "status": "running"},
                ]
            ),
        )
        self.assertEqual(
            [entry["status"] for entry in cache_lst],
            ["finished", "running", "running", "aborted"],
        )

    @unittest.skipIf(
        sys.platform == "win32" or skip_pysqa_test,
        "pysqa module patching not supported on Windows or when pysqa is not installed",
    )
    def test_get_queue_system_cache_data(self):
        with patch(
            "executorlib.task_scheduler.file.spawner_pysqa.QueueAdapter"
        ) as queue_adapter_mock:
            queue_adapter_mock.return_value.get_queue_status.return_value = DataFrame(
                [{"jobid": 21, "status": "pending"}]
            )
            cache_lst = get_queue_system_cache_data(
                cache_dict=[{"filename": "task_i.h5", "queue_id": 21}],
                queue_type="flux",
                config_directory="config_dir",
            )
        queue_adapter_mock.assert_called_once_with(
            directory="config_dir",
            queue_type="flux",
        )
        self.assertEqual(cache_lst[0]["status"], "pending")

    def tearDown(self):
        shutil.rmtree("executorlib_cache", ignore_errors=True)
