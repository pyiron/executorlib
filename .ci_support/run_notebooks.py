"""
Execute the documentation notebooks with papermill for a given job scheduler.

The documentation notebooks demonstrate executorlib for both SLURM and flux, but each CI job only provides one of the
two job schedulers. To keep the rendered documentation free of any CI plumbing, the selection is based on metadata
which is not visible in the rendered notebooks:

Cell tags (cell metadata "tags"):
    "slurm" - the code cell is only executed in the SLURM CI job (notebooks_slurm).
    "flux"  - the code cell is only executed in the flux CI job (notebooks).
    no tag  - the code cell is executed in both jobs, e.g. function definitions or cleaning the cache.

Notebook metadata (top-level notebook metadata, only evaluated for --scheduler slurm):
    "executorlib_ci": {"allocation": true}
        The notebook uses the SlurmJobExecutor (srun) and is executed inside a SLURM allocation, which is created with
        "salloc -n <cores>". Notebooks without this flag are executed on the login node, i.e. directly on the runner,
        which is the setup for the SlurmClusterExecutor (sbatch).
    "executorlib_ci": {"allocation": true, "cores": 2}
        Number of cores requested for the allocation, defaults to 2. The single node test cluster only provides 2 CPU
        cores, so the allocation occupies the whole cluster: a notebook executed inside an allocation must not submit
        jobs with sbatch (SlurmClusterExecutor), as these jobs would wait for the allocation to finish. Notebooks which
        combine both, like a SlurmJobExecutor nested in a function submitted via the SlurmClusterExecutor, are executed
        on the login node.

For each input notebook the cells which do not match the selected scheduler are removed, the filtered notebook is
written to a temporary directory and executed with papermill in the directory of the original notebook. The executed
notebook is stored next to the original notebook as <name>-out.ipynb.

Usage:
    python .ci_support/run_notebooks.py --scheduler slurm notebooks/2-hpc-cluster.ipynb notebooks/story-*.ipynb
    flux start python .ci_support/run_notebooks.py --scheduler flux notebooks/2-hpc-cluster.ipynb
"""

import argparse
import os
import subprocess
import sys
import tempfile

import nbformat

SCHEDULERS = ("flux", "slurm")


def filter_notebook(nb: nbformat.NotebookNode, scheduler: str) -> nbformat.NotebookNode:
    """Remove the code cells which are tagged for a different job scheduler."""
    other_tags = {s for s in SCHEDULERS if s != scheduler}
    nb.cells = [
        cell
        for cell in nb.cells
        if cell.cell_type != "code"
        or not other_tags.intersection(cell.get("metadata", {}).get("tags", []))
    ]
    return nb


def get_command(
    nb: nbformat.NotebookNode,
    scheduler: str,
    input_path: str,
    output_path: str,
    cwd: str,
) -> list[str]:
    """Construct the papermill command, wrapped in a SLURM allocation if the notebook requests it."""
    command = ["papermill", input_path, output_path, "-k", "python3", "--cwd", cwd]
    ci_config = nb.metadata.get("executorlib_ci", {})
    if scheduler == "slurm" and ci_config.get("allocation", False):
        command = ["salloc", "-n", str(ci_config.get("cores", 2))] + command
    return command


def run_notebook(path: str, scheduler: str, tmp_dir: str) -> bool:
    nb = filter_notebook(nb=nbformat.read(path, as_version=4), scheduler=scheduler)
    name = os.path.splitext(os.path.basename(path))[0]
    input_path = os.path.join(tmp_dir, name + "-" + scheduler + ".ipynb")
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(path)), name + "-out.ipynb"
    )
    nbformat.write(nb, input_path)
    command = get_command(
        nb=nb,
        scheduler=scheduler,
        input_path=input_path,
        output_path=output_path,
        cwd=os.path.dirname(os.path.abspath(path)),
    )
    print("Execute:", " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scheduler", choices=SCHEDULERS, required=True)
    parser.add_argument("notebooks", nargs="*")
    args = parser.parse_args()
    if len(args.notebooks) == 0:
        print("No notebooks to execute.")
        return 0
    failed = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        # skip the output of previous executions which is matched by glob patterns like notebooks/story-*.ipynb
        for path in [p for p in args.notebooks if not p.endswith("-out.ipynb")]:
            if not run_notebook(path=path, scheduler=args.scheduler, tmp_dir=tmp_dir):
                failed.append(path)
    if len(failed) > 0:
        print("Failed notebooks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
