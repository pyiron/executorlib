---
title: 'Executorlib – Up-scaling Python workflows for hierarchical heterogenous high-performance computing'
tags:
  - Python
  - High Performance Computing
  - Task Scheduling
authors:
  - name: J. Janssen
    orcid: 0000-0001-9948-7119
    affiliation: "1, 2"
  - name: M.G. Taylor
    orcid: 0000-0003-4327-2746
    affiliation: 2
  - name: P. Yang
    orcid: 0000-0003-4726-2860
    affiliation: 2
  - name: J. Neugebauer
    orcid: 0000-0002-7903-2472
    affiliation: 1
  - name: D. Perez
    orcid: 0000-0003-3028-5249
    affiliation: 2

affiliations:
 - name: Max Planck Institute for Sustainable Materials, Düsseldorf, Germany
   index: 1
 - name: Los Alamos National Laboratory, Los Alamos, NM, United States of America
   index: 2
date: 31 January 2025
bibliography: paper.bib
---

# Summary
Executorlib enables the execution of hierarchical Python workflows on heterogenous computing resources of high-performance computing (HPC) clusters. This is achieved by extending the Executor class of the Python standard library for asynchronously executing callables with an interface to HPC job schedulers. The initial release of Executorlib supports the Simple Linux Utility for Resource Management (SLURM) and the flux framework as HPC job schedulers to start Python processes with dedicated computing resources such as CPU cores, memory or accelerators like GPUs. For heterogenous workflows, Executorlib enables the use of parallel computing frameworks like the message passing interface (MPI) or of dedicated GPU libraries on a per workflow step basis. Python workflows can be up-scaled with Executorlib from a laptop up to the latest Exascale HPC clusters with minimal code changes including support for hierarchical workflows. 

# Statement of Need
The convergence of artificial intelligence (AI) and high-performance computing (HPC) workflows [@workflows] is one of the key drivers for the rise of Python workflows for HPC. To avoid intrusive code changes, interfaces to performance critical scientific software packages were traditionally implemented using file-based communication and control shell scripts, leading to poor maintainability, portability, and scalability. This approach is however losing ground to more efficient alternatives, such as the use of direct Python bindings, as their support is now increasingly common in scientific software packages and especially machine learning packages and AI frameworks. This enables the programmer to easily express complex workloads that require the orchestration of multiple codes. Still, Python workflows for HPC also come with challenges, like (1) safely terminating Python processes, (2) controlling the resources of Python processes and (3) the management of Python environments [@pythonhpc]. The first two of these challenges can be addressed by developing strategies and tools to interface HPC job schedulers such as SLURM [@slurm] with Python in order to control the execution and manage the computational resources required to execute heterogenous HPC workflows. A number of Python workflow frameworks have been developed for both types of interfaces, ranging from domain-specific solutions for fields like high-throughput screening in computational materials science, i.e. fireworks [@fireworks], pyiron [@pyiron] and aiida [@aiida], to generalized Python interfaces for job schedulers, i.e. myqueue [@myqueue] and PSI/j [@psij] and task scheduling frameworks which implement their own task scheduling on top of the HPC job scheduler, i.e. dask [@dask], parsl [@parsl] and jobflow [@jobflow]. While these tools can be powerful, they introduce new constructs that unfamiliar to most Python developers, adding complexity and creating a barrier to entry.

# Features and Implementation
To address this limitation while at the same time leveraging the powerful and novel hierarchical HPC resource managers like the flux framework [@flux], we introduce Executorlib, which instead leverages and naturally extends the familiar Executor interface defined by the Python standard library from single-node shared-memory operation to multi-node distributed operation on HPC platforms. \autoref{fig:process} illustrates the internal functionality of Executorlib.

![Illustration of the communication between the Executorlib Executor, the job scheduler and the Python process to asynchronously execute the submitted Python function (on the right).\label{fig:process}](process.png){width="50%"}

Rather than implementing its own job scheduler, Executorlib instead leverages existing job schedulers to request and manage Python processes and associated computing resources. Further, instead of defining a new syntax and concepts, Executorlib extends the existing syntax of the Executor class in the Python standard library. Currently, Executorlib supports five different job schedulers implement as different Executor classes. The first is the `SingleNodeExecutor` for rapid prototyping on a laptop or local workstation in a way that is functionally similar to the standard `ProcessPoolExecutor`. The second, `SlurmClusterExecutor` submits Python functions as individual jobs to a SLURM job scheduler using the `sbatch` command, which can be useful for long-running tasks, e.g., that call a compute intensive legacy code. The third is the `SlurmJobExecutor` which distributes Python functions in an existing SLURM job using the `srun` command. In analogy, the `FluxClusterExecutor` submits Python functions as individual jobs to a flux job scheduler and the `FluxJobExecutor` distributes Python functions in a flux job. Given the hierarchial approach of the flux scheduler there is no limit to the number of `FluxJobExecutor` instances which can be nested inside each other to construct hierarchical workflows. 

To assign dedicated computing resources to individual Python functions, the Executorlib Executor classes extend the submission function `submit()` to support not only the Python function and its inputs, but also a Python dictionary specifying the requested computing resources `resource_dict`. The resource dictionary can define the number of compute cores, number of threads, number of GPUs, as well as job scheduler specific parameters like the working directory, maximum run time or the maximum memory. With this hierarchical approach, Executorlib allows the user to finely control the execution of each individual Python function, using parallel communication libraries like the Message Passing Interface (MPI) for Python [@mpi4py] or GPU-optimized libraries to aggressively optimize complex compute intensive tasks of heterogenous HPC that are best solved by tightly-coupled parallelization approaches, while offering a simple and easy to maintain approach to the orchestration of many such weakly-coupled tasks. This ability to seamlessly combine different programming models again accelerates the rapid prototyping of heterogenous HPC workflows without sacrificing performance of critical code components.

# Usage To-Date 
While initially developed in the US DOE Exascale Computing Project’s Exascale Atomistic Capability for Accuracy, Length and Time (EXAALT) to accelerate the development of computational materials science simulation workflows for the Exascale, Executorlib has since been generalized to support a wide-range of backends and HPC clusters at different scales. Based on this generalization, it is also been implemented in the pyiron workflow framework [@pyiron] as primary task scheduling interface. 

# Additional Details 
The full documentation including a number of examples for the individual features is available at [executorlib.readthedocs.io](https://executorlib.readthedocs.io) and the corresponding source code at [github.com/pyiron/executorlib](https://github.com/pyiron/executorlib). Executorlib is developed an as open-source library with a focus on stability. 

# Acknowledgements
J.J. was supported by the Laboratory Directed Research and Development program of Los Alamos National Laboratory under project number 20220815PRD4. J.J. and D.P. acknowledge funding from the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration, and the hospitality from the “Data-Driven Materials Informatics” program from the Institute of Mathematical and Statistical Innovation (IMSI). J.J. and J.N. acknowledge funding from the Deutsche Forschungsgemeinschaft (DFG) through the CRC1394 “Structural and Chemical Atomic Complexity – From Defect Phase Diagrams to Material Properties”, project ID 409476157. J.J., M.G.T., P.Y., J.N. and D.P. acknowledge the hospitality of the Institute of Pure and Applied math (IPAM) as part of the “New Mathematics for the Exascale: Applications to Materials Science” long program. M.G.T. and P.Y. acknowledge support of the U.S. Department of Energy (DOE), Office of Science, Office of Basic Energy Sciences, Heavy Element Chemistry Program (KC0302031) under contract number E3M2. Los Alamos National Laboratory is operated by Triad National Security LLC, for the National Nuclear Security administration of the U.S. DOE under Contract No. 89233218CNA0000001.

# References
