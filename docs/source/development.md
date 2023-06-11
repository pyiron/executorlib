# Development
The `pympipool` package is developed based on the need to simplify the up-scaling of python functions over multiple 
compute nodes. The project is under active development, so the difference between the individual interfaces might not 
always be clearly defined. The `pympipool.Pool` interface is the oldest and consequently currently most stable but at 
the same time also most limited interface. The `pympipool.Executor` is the recommended interface for most workflows but
it can be computationally less efficient than the `pympipool.PoolExecutor` interface for large number of serial python
functions. Finally, the `pympipool.MPISpawnPool` is primarily a prototype of an alternative interface, which is available
for testing but typically not recommended, based on the limitations of initiating new communicators.

Any feedback and contributions are welcome. 