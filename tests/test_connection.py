import os
import unittest
from pympipool.shared.connections import BaseInterface


class Interface(BaseInterface):
    def __init__(self, cwd, cores=1, gpus_per_core=0, oversubscribe=False):
        super().__init__(
            cwd=cwd,
            cores=cores,
            gpus_per_core=gpus_per_core,
            oversubscribe=oversubscribe,
        )


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.interface = Interface(
            cwd=os.path.abspath("."),
            cores=1,
            gpus_per_core=0,
            oversubscribe=False
        )

    def test_bootup(self):
        with self.assertRaises(NotImplementedError):
            self.interface.bootup(command_lst=[])

    def test_shutdown(self):
        with self.assertRaises(NotImplementedError):
            self.interface.shutdown(wait=True)

    def test_poll(self):
        with self.assertRaises(NotImplementedError):
            self.interface.poll()
