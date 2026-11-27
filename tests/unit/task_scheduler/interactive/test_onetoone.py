import unittest
from unittest.mock import patch

from executorlib import SingleNodeExecutor
from executorlib.task_scheduler.interactive.onetoone import OneProcessTaskScheduler


def _identity(x):
    return x


class TestOneProcessTaskScheduler(unittest.TestCase):
    def test_submit_rejects_cores_exceeding_max_cores(self):
        with patch.object(OneProcessTaskScheduler, "_set_process"):
            scheduler = OneProcessTaskScheduler(max_cores=2)
            with self.assertRaises(ValueError):
                scheduler.submit(_identity, 1, resource_dict={"cores": 4})

    def test_submit_accepts_cores_within_max_cores(self):
        with patch.object(OneProcessTaskScheduler, "_set_process"):
            scheduler = OneProcessTaskScheduler(max_cores=2)
            future = scheduler.submit(_identity, 1, resource_dict={"cores": 2})
            self.assertFalse(future.done())

    def test_single_node_executor_disable_dependencies_rejects_excess_cores(self):
        with patch.object(OneProcessTaskScheduler, "_set_process"):
            executor = SingleNodeExecutor(max_cores=2, disable_dependencies=True)
            with self.assertRaises(ValueError):
                executor.submit(_identity, 1, resource_dict={"cores": 4})
