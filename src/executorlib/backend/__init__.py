"""
Backend for executorlib, these are the executables called by executorlib to initialize the Python processes which
receive the Python functions for execution. The backends are divided based on two criteria, once whether they are using
the file based cache as it is employed by the FluxClusterExecutor and SlurmClusterExecutor versus the interactive
interface and secondly whether the submitted Python function is executed MPI parallel or in serial.
"""
