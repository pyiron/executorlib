# Technical Concepts

The `executorlib` package is designed to up-scale Python functions for High Performance Computing (HPC) by extending the standard Python `Executor` interface. This document explains the underlying technical concepts and the internal architecture of `executorlib`.

## Internal Architecture

The `executorlib` library is structured into four primary modules:

*   **`executor`**: Defines the user-facing `Executor` classes (e.g., `SingleNodeExecutor`, `SlurmClusterExecutor`, `SlurmJobExecutor`, `FluxClusterExecutor`, `FluxJobExecutor`). These classes provide the primary interface for users to submit tasks.
*   **`task_scheduler`**: Manages the distribution and scheduling of tasks. It handles task queues, resource allocation, and coordinates with spawners.
*   **`standalone`**: Contains utility functions and classes that do not depend on other internal modules. This includes serialization (using `cloudpickle`), ZMQ-based communication (`SocketInterface`), and input validation.
*   **`backend`**: Contains the code executed by the worker processes to perform the actual function calls.

## Class Hierarchy and Coupling

The following diagram illustrates the relationship between the main classes in `executorlib`.

```{mermaid}
classDiagram
    class FutureExecutor {
        <<interface>>
    }
    class BaseExecutor {
        -_task_scheduler: TaskSchedulerBase
        +submit(fn, *args, **kwargs) Future
        +shutdown(wait)
    }
    class TaskSchedulerBase {
        -_future_queue: Queue
        -_process: Thread
        +submit(fn, *args, **kwargs) Future
    }
    class BaseSpawner {
        <<interface>>
        +bootup(command_lst)
        +shutdown(wait)
    }
    class SocketInterface {
        +send_dict(input_dict)
        +receive_dict() dict
    }

    FutureExecutor <|-- BaseExecutor
    BaseExecutor o-- TaskSchedulerBase
    TaskSchedulerBase <|-- OneProcessTaskScheduler
    TaskSchedulerBase <|-- BlockAllocationTaskScheduler
    TaskSchedulerBase <|-- DependencyTaskScheduler
    TaskSchedulerBase <|-- FileTaskScheduler

    OneProcessTaskScheduler o-- BaseSpawner
    BaseSpawner <|-- MpiExecSpawner
    BaseSpawner <|-- SrunSpawner
    BaseSpawner <|-- FluxPythonSpawner

    OneProcessTaskScheduler ..> SocketInterface : uses
```

## Execution Flow

When a user submits a function to an executor, several steps occur in the background to ensure the task is executed with the requested resources and the result is returned.

```{mermaid}
sequenceDiagram
    participant User
    participant Executor
    participant TaskScheduler
    participant Spawner
    participant Backend

    User->>Executor: submit(fn, args, resource_dict)
    Executor->>TaskScheduler: submit(fn, args, resource_dict)
    TaskScheduler->>TaskScheduler: Add to _future_queue
    TaskScheduler-->>User: Return Future object

    Note over TaskScheduler, Spawner: Task loop in background thread

    TaskScheduler->>Spawner: bootup(command)
    Spawner->>Backend: Start worker process
    TaskScheduler->>Backend: Send function and arguments (ZMQ/File)
    Backend->>Backend: Execute function
    Backend->>TaskScheduler: Send result (ZMQ/File)
    TaskScheduler->>User: Update Future with result
```

## Communication Modes

`executorlib` supports two primary communication modes between the main process and the worker processes:

### Interactive Communication (ZMQ-based)
Used by `SingleNodeExecutor`, `SlurmJobExecutor`, and `FluxJobExecutor`. It leverages [ZeroMQ (ZMQ)](https://zeromq.org) and [cloudpickle](https://github.com/cloudpipe/cloudpickle) for high-performance, in-memory communication of Python objects. This mode is ideal for low-latency task distribution within an allocation.

### File-based Communication
Used by `SlurmClusterExecutor` and `FluxClusterExecutor`. It uses the filesystem to communicate between the main process and the individual HPC jobs. This mode is necessary when tasks are submitted as independent jobs to a scheduler like SLURM or Flux, where direct network communication between the login node and compute nodes might be restricted.

## Resource Management

One of the key features of `executorlib` is the ability to specify resources on a per-function-call basis using the `resource_dict`.

*   **`cores`**: Number of MPI ranks or CPU cores.
*   **`threads_per_core`**: Number of OpenMP threads.
*   **`gpus_per_core`**: Number of GPUs.
*   **`cwd`**: Working directory for the task.

The `TaskScheduler` ensures that these resource requirements are translated into appropriate commands for the `Spawner` (e.g., `mpiexec`, `srun`, or `flux run`).
