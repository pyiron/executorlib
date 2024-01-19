# Development
The `pympipool` package is developed based on the need to simplify the up-scaling of python functions over multiple 
compute nodes. The project is under active development, so the difference between the individual interfaces might not 
always be clearly defined. 

## Contributions 
Any feedback and contributions are welcome. 

## Integration
The key functionality of the `pympipool` package is the up-scaling of python functions with thread based parallelism, 
MPI based parallelism or by assigning GPUs to individual python functions. In the background this is realized using a 
combination of the [zero message queue](https://zeromq.org) and [cloudpickle](https://github.com/cloudpipe/cloudpickle) 
to communicate binary python objects. The `pympipool.communication.SocketInterface` is an abstraction of this interface,
which is used in the other classes inside `pympipool` and might also be helpful for other projects. It comes with a 
series of utility functions:

* `pympipool.communication.interface_bootup()`: To initialize the interface
* `pympipool.communication.interface_connect()`: To connect the interface to another instance
* `pympipool.communication.interface_send()`: To send messages via this interface 
* `pympipool.communication.interface_receive()`: To receive messages via this interface 
* `pympipool.communication.interface_shutdown()`: To shutdown the interface

## Alternative Projects
[dask](https://www.dask.org), [fireworks](https://materialsproject.github.io/fireworks/) and [parsl](http://parsl-project.org)
address similar challenges. On the one hand they are more restrictive when it comes to the assignment of resource to 
a given worker for execution, on the other hand they provide support beyond the high performance computing (HPC)
environment. 