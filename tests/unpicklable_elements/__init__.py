"""
`pympipool` should be able to handle the case where _no_ elements of the execution can
be pickled with the traditional `pickle` module but rather require `cloudpickle`.

This is particularly important for compatibility with `pyiron_workflow`, which
dynamically defines (unpickleable) all sorts of objects.

Currently, `pyiron_workflow` defines its own executor,
`pyiron_workflow.executors.CloudPickleProcessPool`, which can handle these unpickleable
 things, but is otherwise very primitive compared to
 `pympipool.mpi.executor.PyMPISingleTaskExecutor`.

Simply replacing `CloudPickleProcessPool` with `PyMPISingleTaskExecutor` in the
`pyiron_atomistics` tests mostly works OK, and work perfectly when the tests are ported
to a notebook, but some tests hang indefinitely on CI and running unittests locally.

To debug this, we break the tests up into their individual components (so hanging
doesn't stop us from seeing the results of other tests). Once everything is running,
these can be re-condensed into a single test file and this entire subdirectory can be
deleted.
"""
