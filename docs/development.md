# Development
The `pympipool` package is developed based on the need to simplify the up-scaling of python functions over multiple 
compute nodes. The project is used for Exascale simualtion in the context of computational chemistry and materials 
science. Still it remains a scientific research project with the goal to maximize the utilization of computational 
resources for scientific applications. No formal support is provided. 

## Contributions 
Any feedback and contributions are welcome. 

## License
```
BSD 3-Clause License

Copyright (c) 2022, Jan Janssen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

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

While `pympipool` was initially designed for up-scaling python functions for HPC, the same functionality can be leveraged
to up-scale any executable independent of the programming language it is developed in. This approach follows the design 
of the `flux.job.FluxExecutor` included in the [flux framework](https://flux-framework.org). In `pympipool` this approach
is extended to support any kind of subprocess, so it is no longer limited to the [flux framework](https://flux-framework.org).

### Subprocess
Following the [`subprocess.check_output()`](https://docs.python.org/3/library/subprocess.html) interface of the standard
python libraries, any kind of command can be submitted to the `pympipool.SubprocessExecutor`. The command can either be 
specified as a list `["echo", "test"]` in which the first entry is typically the executable followed by the corresponding
parameters or the command can be specified as a string `"echo test"` with the additional parameter `shell=True`.
```python
from pympipool import SubprocessExecutor

with SubprocessExecutor(max_workers=2) as exe:
    future = exe.submit(["echo", "test"], universal_newlines=True)
    print(future.done(), future.result(), future.done())
```
```
>>> (False, "test", True)
```
In analogy to the previous examples the `SubprocessExecutor` class is directly imported from the `pympipool` module and 
the maximum number of workers is set to two `max_workers=2`. In contrast to the `pympipool.Executor` class no other
settings to assign specific hardware to the command via the python interface are available in the `SubprocessExecutor` 
class. To specify the hardware requirements for the individual commands, the user has to manually assign the resources
using the commands of the resource schedulers like `srun`, `flux run` or `mpiexec`.

The `concurrent.futures.Future` object returned after submitting a command to the `pymipool.SubprocessExecutor` behaves
just like any other future object. It provides a `done()` function to check if the execution completed as well as a 
`result()` function to return the output of the submitted command. 

In comparison to the `flux.job.FluxExecutor` included in the [flux framework](https://flux-framework.org) the 
`pymipool.SubprocessExecutor` differs in two ways. One the `pymipool.SubprocessExecutor` does not provide any option for
resource assignment and two the `pymipool.SubprocessExecutor` returns the output of the command rather than just 
returning the exit status when calling `result()`. 

### Interactive Shell
Beyond external executables which are called once with a set of input parameters and or input files and return one set
of outputs, there are some executables which allow the user to interact with the executable during the execution. The 
challenge of interfacing a python process with such an interactive executable is to identify when the executable is ready
to receive the next input. A very basis example for an interactive executable is a script which counts to the number 
input by the user. This can be written in python as `count.py`:
```python
def count(iterations):
    for i in range(int(iterations)):
        print(i)
    print("done")


if __name__ == "__main__":
    while True:
        user_input = input()
        if "shutdown" in user_input:
            break
        else:
            count(iterations=int(user_input))
```
This example is challenging in terms of interfacing it with a python process as the length of the output changes depending
on the input. The first option that the `pympipool.ShellExecutor` provides is specifying the number of lines to read for
each call submitted to the executable using the `lines_to_read` parameter. In comparison to the `SubprocessExecutor` 
defined above the `ShellExecutor` only supports the execution of a single executable at a time, correspondingly the input
parameters for calling the executable are provided at the time of initialization of the `ShellExecutor` and the inputs 
are submitted using the `submit()` function:
```python
from pympipool import ShellExecutor

with ShellExecutor(["python", "count.py"], universal_newlines=True) as exe:
    future_lines = exe.submit(string_input="4", lines_to_read=5)
    print(future_lines.done(), future_lines.result(), future_lines.done())
```
```
>>> (False, "0\n1\n2\n3\ndone\n", True)
```
The response for a given set of input is again returned as `concurrent.futures.Future` object, this allows the user to
execute other steps on the python side while waiting for the completion of the external executable. In this case the 
example counts the numbers from `0` to `3` and prints each of them in one line followed by `done` to notify the user its
waiting for new inputs. This results in `n+1` lines of output for the input of `n`. Still predicting the number of lines
for a given input can be challenging, so the `pympipool.ShellExecutor` class also provides the option to wait until a 
specific pattern is found in the output using the `stop_read_pattern`:
```python
from pympipool import ShellExecutor

with ShellExecutor(["python", "count.py"], universal_newlines=True) as exe:
    future_pattern = exe.submit(string_input="4", stop_read_pattern="done")
    print(future_pattern.done(), future_pattern.result(), future_pattern.done())
```
```
>>> (False, "0\n1\n2\n3\ndone\n", True)
```
In this example the pattern simply searches for the string `done` in the output of the program and returns all the output
gathered from the executable since the last input as the result of the `concurrent.futures.Future` object returned after
the submission of the interactive command.
