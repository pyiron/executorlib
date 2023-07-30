import os
import unittest
from pympipool.shared.connections import (
    BaseInterface,
    MpiExecInterface,
    SlurmSubprocessInterface,
    PysqaInterface,
    FluxCmdInterface,
    get_connection_interface
)


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
            cwd=None,
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


class TestInterfaceConnection(unittest.TestCase):
    def test_mpiexec(self):
        interface = get_connection_interface(
            cwd=None,
            cores=1,
            gpus_per_core=0,
            oversubscribe=False,
            enable_flux_backend=False,
            enable_slurm_backend=False,
            queue_adapter=None,
            queue_type=None,
            queue_adapter_kwargs=None,
        )
        self.assertIsInstance(interface, MpiExecInterface)

    def test_slurm(self):
        interface = get_connection_interface(
            cwd=None,
            cores=1,
            gpus_per_core=0,
            oversubscribe=False,
            enable_flux_backend=False,
            enable_slurm_backend=True,
            queue_adapter=None,
            queue_type=None,
            queue_adapter_kwargs=None,
        )
        self.assertIsInstance(interface, SlurmSubprocessInterface)

    def test_pysqa(self):
        interface = get_connection_interface(
            cwd=None,
            cores=1,
            gpus_per_core=0,
            oversubscribe=False,
            enable_flux_backend=False,
            enable_slurm_backend=False,
            queue_adapter=True,
            queue_type=None,
            queue_adapter_kwargs=None,
        )
        self.assertIsInstance(interface, PysqaInterface)

    def test_flux_cmd(self):
        interface = get_connection_interface(
            cwd=None,
            cores=1,
            gpus_per_core=0,
            oversubscribe=False,
            enable_flux_backend=True,
            enable_slurm_backend=False,
            queue_adapter=None,
            queue_type=None,
            queue_adapter_kwargs=None,
        )
        self.assertIsInstance(interface, FluxCmdInterface)
