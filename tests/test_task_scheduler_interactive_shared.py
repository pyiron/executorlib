import shutil
from concurrent.futures import Future
from unittest import TestCase

from executorlib.standalone.command import get_interactive_execute_command
from executorlib.standalone.interactive.communication import interface_bootup, ExecutorlibSocketError
from executorlib.standalone.interactive.spawner import SubprocessSpawner
from executorlib.task_scheduler.interactive.shared import execute_task_dict


def get_error():
    raise ExecutorlibSocketError()


class TestExecuteTaskDictWithoutCache(TestCase):
    def test_execute_task_sum(self):
        f = Future()
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertFalse(f.done())
        result = execute_task_dict(
            task_dict={"fn": sum, "args": ([1, 2], ), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory=None,
            cache_key=None,
            error_log_file=None,
        )
        self.assertTrue(result)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 3)

    def test_execute_task_done(self):
        f = Future()
        f.set_result(5)
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertTrue(f.done())
        result = execute_task_dict(
            task_dict={"fn": sum, "args": ([1, 2], ), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory=None,
            cache_key=None,
            error_log_file=None,
        )
        self.assertTrue(result)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 5)

    def test_execute_task_error(self):
        f = Future()
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertFalse(f.done())
        result = execute_task_dict(
            task_dict={"fn": get_error, "args": (), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory=None,
            cache_key=None,
            error_log_file=None,
        )
        self.assertFalse(result)
        self.assertFalse(f.done())


class TestExecuteTaskDictWithCache(TestCase):
    def tearDown(self):
        shutil.rmtree("cache_execute_task", ignore_errors=True)

    def test_execute_task_sum(self):
        f = Future()
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertFalse(f.done())
        result = execute_task_dict(
            task_dict={"fn": sum, "args": ([1, 2], ), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory="cache_execute_task",
            cache_key=None,
            error_log_file=None,
        )
        self.assertTrue(result)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 3)

    def test_execute_task_done(self):
        f = Future()
        f.set_result(5)
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertTrue(f.done())
        result = execute_task_dict(
            task_dict={"fn": sum, "args": ([1, 2], ), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory="cache_execute_task",
            cache_key=None,
            error_log_file=None,
        )
        self.assertTrue(result)
        self.assertTrue(f.done())
        self.assertEqual(f.result(), 5)

    def test_execute_task_error(self):
        f = Future()
        interface, success_flag = interface_bootup(
            command_lst=get_interactive_execute_command(
                cores=1,
            ),
            connections=SubprocessSpawner(),
            hostname_localhost=True,
            log_obj_size=False,
            worker_id=1,
            stop_function=None,
        )
        self.assertTrue(success_flag)
        self.assertFalse(f.done())
        result = execute_task_dict(
            task_dict={"fn": get_error, "args": (), "kwargs": {}},
            future_obj=f,
            interface=interface,
            cache_directory="cache_execute_task",
            cache_key=None,
            error_log_file=None,
        )
        self.assertFalse(result)
        self.assertFalse(f.done())