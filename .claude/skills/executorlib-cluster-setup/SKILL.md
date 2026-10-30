---
name: executorlib-cluster-setup
description: Analyse a SLURM cluster with its command-line tools (sinfo, scontrol, sacctmgr, sbatch --test-only) and generate a cluster-specific executorlib skill (submission template, resource_dict helpers, tested limits and pitfalls) for submitting Python functions with SlurmClusterExecutor. Use when a user wants to set up executorlib for a new or not yet configured cluster, asks to "create an executorlib skill for <cluster>", or when no executorlib-<cluster> skill exists for the machine the session runs on.
---

# Generating an executorlib skill for a SLURM cluster

Goal: a skill `executorlib-<cluster>/SKILL.md` like `executorlib-raven`, `executorlib-cmti` and `executorlib-cmmg`. Reuse their structure. Each fact in the new skill must come from a command run on the cluster, from the site documentation, or from a job that actually ran. Write "not tested" next to anything else.

Work on the cluster's login node. Ask the user only about things the cluster can't tell you: which codes they run (MPI binaries, GPU codes, plain Python), which project/account to charge, and the site documentation URL.

## Step 1: Partitions and nodes

```bash
sinfo -s                                              # partitions, time limits, node counts; '*' = default partition
sinfo -o "%P %a %l %D %c %m %G %f"                    # per node group: CPUs, memory (MB), GRES (GPUs), features
scontrol show partition <p> | tr ' ' '\n' | grep -E "^(MaxNodes|MinNodes|MaxTime|DefMemPerCPU|DefMemPerNode|MaxMemPerNode|MaxMemPerCPU|OverSubscribe|MaxCPUsPerNode|QoS)="
scontrol show node <one node per group> | grep -E "CPUTot|Sockets|CoresPerSocket|ThreadsPerCore|RealMemory|Gres|AvailableFeatures"
```

Run `scontrol show partition` once per partition: passing several names to one call printed nothing on Raven. Record, for each partition:
- **shared or exclusive** (`OverSubscribe=EXCLUSIVE` means whole nodes, as on `general` / `p.cmfe`);
- **single- or multi-node** (`MaxNodes=1` means single node, as on `small` / `s.cmfe`);
- **default memory** (`DefMemPerCPU` / `DefMemPerNode`). A low default, such as 1.5 GB per CPU, means the template needs `--mem`;
- **physical cores vs logical CPUs** (`ThreadsPerCore=2` means `CPUTot` counts SMT threads: Raven reports 144 for 72 cores). By default, plan MPI ranks per physical core;
- **GPUs**: GRES name and count (`gpu:a100:4`), and the features needed to select GPU nodes (`--constraint=gpu`);
- **test/debug partitions** with short limits (`gpudev`, 15 min). These are the fastest way to validate the setup.

## Step 2: Policy and submit filter

```bash
scontrol show config | grep -iE "JobSubmitPlugins|SelectTypeParameters|MpiDefault|DefMemPerCPU"
sacctmgr -nP show assoc user=$USER format=cluster,account,partition,qos,maxsubmit,maxjobs
sacctmgr -nP show qos format=name,maxwall,maxsubmitpu,maxjobspu,maxtrespu
srun --mpi=list                                       # pmix / pmi2 availability for `srun --mpi=...`
```

- `JobSubmitPlugins = lua` means a site submit filter that can choose partitions or reject resource combinations. Find its rules in the site documentation and confirm them with `sbatch --test-only` (Step 4). On Raven it picks the GPU partition by itself and rejects 1-GPU jobs that ask for more than 1/4 of the node's memory.
- `maxsubmit` / `maxjobs` limit how many executorlib tasks can be queued at once. A failed `sbatch` is stored in the executorlib cache as an error (see the pitfalls).
- `CR_ONE_TASK_PER_CORE` together with SMT means tasks are placed per physical core.

## Step 3: Python environment

```bash
python -c "import executorlib, pysqa; print(executorlib.__version__, executorlib.__file__, pysqa.__version__)"
grep -n "dependency_list\|threads_per_core\|unsupported_keys" -A3 $(python -c "import executorlib, os; print(os.path.dirname(executorlib.__file__))")/task_scheduler/file/spawner_pysqa.py
module avail 2>&1 | head; which srun mpirun
```

Read `spawner_pysqa.py`. It shows exactly which keys reach the Jinja2 template, and this varies between executorlib versions:
- `command`, `working_directory`, `run_time_max` (seconds), `job_name`, and the dependencies as **`dependency_list`**. Older skills used `dependency`, which never rendered;
- every other `resource_dict` key, including custom ones such as `partition`, `nodes`, `gpus_per_node`, `cpus_per_task`, `memory_max` and `mpi_ranks`;
- `cores` is multiplied by `threads_per_core`, and `gpus_per_core` / `slurm_cmd_args` / `openmpi_oversubscribe` are dropped. Therefore request GPUs and threads through custom keys.

Also note where the Python that runs on the compute nodes is installed, and any module environment that compiled binaries need (for example `source ~/lammps/raven_env.sh`).

## Step 4: Write the template and test it without running anything

Start from this skeleton and add only the lines the cluster needs (`--partition`, `--mem`, `--constraint`, `--gres`, `--exclusive`, module `source`, `OMP_*`):

```python
SUBMISSION_TEMPLATE = """\
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env
{%- if partition %}
#SBATCH --partition={{partition}}
{%- endif %}
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if dependency_list %}
#SBATCH --dependency=afterok:{{ dependency_list | join(',') }}
{%- endif %}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks-per-node={{ntasks_per_node}}
#SBATCH --cpus-per-task={{cpus_per_task}}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}
{%- endif %}

{{command}}
"""
```

Add a `resource_dict(...)` helper that encodes the cluster's rules (cores and memory per GPU, ranks per node, when to switch to multi-node), always with `"cores": 1` and an optional `cache_key`.

Render every resource combination and pass it to `sbatch --test-only`. This shows syntax errors and submit-filter rejections immediately, without queueing a job:

```python
import subprocess
from jinja2 import Template


def test_only(rd, path="test_job.sh"):
    script = Template(rd["submission_template"]).render(
        job_name="probe",
        working_directory=os.getcwd(),
        command="hostname",
        dependency_list=[],
        **{k: v for k, v in rd.items() if k != "submission_template"},
    )
    open(path, "w").write(script)
    return subprocess.run(
        ["sbatch", "--test-only", path], capture_output=True, text=True
    ).stderr
```

Examples of what this found on Raven: `--get-user-env=L` (the pysqa default) is rejected ("doesn't allow an argument"), and a 1-GPU job with too much `--mem` is rejected by the Lua filter.

## Step 5: Smoke tests on the compute nodes

Use the debug/test partition when there is one. Give every task an explicit `cache_key`, and use a fresh cache directory such as `executorlib_cache_setup`:

1. **Serial function:** return `socket.gethostname()`, `len(os.sched_getaffinity(0))`, the `SLURM_*` variables and `OMP_NUM_THREADS`. This checks that the CPU count matches `cpus_per_task`.
2. **GPUs** (if any): `CUDA_VISIBLE_DEVICES`, `nvidia-smi -L`, and a small matmul per device (CuPy/PyTorch) for 1 and for all GPUs of a node.
3. **MPI binary launched from the function** (if the user runs one): the function calls `srun --mpi=<pmix|pmi2> -n <ranks> <binary>`, with `"cores": 1`. Check 1 rank against N ranks on one node, then against 2 nodes (same energy/result). The CMTI skill verified 80 ranks on 2 nodes this way.
4. **Dependency:** submit `b = exe.submit(g, a)` where `a` is the future of a job that sleeps. `squeue` must show `b` as `(Dependency)`, and `run_queue.sh` in its cache folder must contain `--dependency=afterok:<id>`.
5. **Detach and collect:** submit, wait for the queue ids, `exe.shutdown(wait=False)`, then read the result in a new kernel with `executorlib.get_cache_data(...)`.
6. Optional: a short **scaling** run of the user's main code (throughput vs ranks/GPUs). This gives the per-node numbers the skill should quote.

Read `<cache>/<cache_key>/run_queue.sh`, `time.out` and `error.out` after every test.

## Step 6: Write `executorlib-<cluster>/SKILL.md`

Put it next to the other skills (`~/.claude/skills/executorlib-<cluster>/SKILL.md`, or the user's skills repo) and use these sections:

1. **Frontmatter:** `name: executorlib-<cluster>` and a `description` naming the cluster, its hardware, its partitions, and the trigger phrases ("run on <cluster>", partition names).
2. **Intro and worked examples:** the paths of the notebooks used for the tests.
3. **Cluster facts:** a table of partition / use / nodes / limits, then bullets for cores, memory, GPUs, the submit filter and job limits, each with its source.
4. **Submission template(s):** one per job type (CPU, GPU, multi-node MPI) and the `resource_dict` helpers.
5. **Patterns:** serial function; function that launches an MPI or GPU binary (`cores: 1`, ranks in the template); many jobs (cache keys, `remove_failed`, `wait_for_queue_ids`, detach); chained restart segments; `block_allocation` only if tested.
6. **Pitfalls:** the general ones below, plus everything found in Steps 4-5.
7. **Monitoring and files:** the cache layout, `squeue`, and the install locations.

Copy the generic helpers (`wait_for_queue_ids`, `cache_entries`, `remove_failed`, `cached_results`) from `executorlib-raven` instead of rewriting them.

## General pitfalls to carry into every cluster skill

- `resource_dict["cores"] > 1` starts that many copies of the Python function (it is meant for mpi4py functions). A function that calls `srun` itself must be submitted with `cores: 1`, otherwise the nested `srun` calls deadlock.
- The default cache key hashes the pickled function, including the `/tmp/ipykernel_<pid>` path of a notebook, so it changes between kernels. **Always set `cache_key`.** The key ignores `resource_dict`.
- Failed submissions and crashed jobs are cached as errors and returned again on the next run. Delete them before resubmitting.
- Workers are cloudpickled and run where the notebook folder is not on `sys.path`. Keep them self-contained (imports inside the function, absolute paths), or use `cloudpickle.register_pickle_by_value(module)`. Test with `python -I` on an unpickled task before submitting.
- Don't use `threads_per_core` / `gpus_per_core` for SLURM CPU/GPU binding. Use custom template keys.
- executorlib's warning `The following keys are not recognized and cannot be validated` about custom keys is expected.