# Trouble shooting

## When `flux` fails:

### Step-by-Step Guide to Create a Custom Jupyter Kernel for Flux

#### Step 1: Create a New Kernel Specification

1. Install [`flux-core`](https://anaconda.org/conda-forge/flux-core) in your Jupyter environment:

   ```bash
   conda install -c conda-forge flux-core
   ```

2. **Find the Jupyter Kernel Directory**:

   Open your terminal or command prompt and run:

   ```bash
   jupyter --paths
   ```

   This command will display the paths where Jupyter looks for kernels. You'll usually find a directory named `kernels` under the `jupyter` data directory. You will create a new directory for the Flux kernel in the `kernels` directory.

3. **Create the Kernel Directory**:

   Navigate to the kernels directory (e.g., `~/.local/share/jupyter/kernels` on Linux or macOS) and create a new directory called `flux`.

   ```bash
   mkdir -p ~/.local/share/jupyter/kernels/flux
   ```

   If you're using Windows, the path will be different, such as `C:\Users\<YourUsername>\AppData\Roaming\jupyter\kernels`.

4. **Create the `kernel.json` File**:

   Inside the new `flux` directory, create a file named `kernel.json`:

   ```bash
   nano ~/.local/share/jupyter/kernels/flux/kernel.json
   ```

   Paste the following content into the file:

   ```json
   {
     "argv": [
       "flux",
       "start",
       "/srv/conda/envs/notebook/bin/python",
       "-m",
       "ipykernel_launcher",
       "-f",
       "{connection_file}"
     ],
     "display_name": "Flux",
     "language": "python",
     "metadata": {
       "debugger": true
     }
   }
   ```

   - **`argv`**: This array specifies the command to start the Jupyter kernel. It uses `flux start` to launch Python in the Flux environment.
   - **`display_name`**: The name displayed in Jupyter when selecting the kernel.
   - **`language`**: The programming language (`python`).

   **Note**:

   - Make sure to replace `"/srv/conda/envs/notebook/bin/python"` with the correct path to your Python executable. You can find this by running `which python` or `where python` in your terminal.
   - If you installed `flux` in a specific environment, you have to write the absolute path to `flux` in the `argv` array.

#### Step 2: Restart Jupyter Notebook

1. **Restart the Jupyter Notebook Server**:

   Close the current Jupyter Notebook server and restart it:

   ```bash
   jupyter notebook
   ```

   ```bash
   jupyter lab
   ```

    Or simply restart your server.

2. **Select the Flux Kernel**:

   When creating a new notebook or changing the kernel of an existing one, you should see an option for "Flux" in the list of available kernels. Select it to run your code with the Flux environment.

#### Step 3: Run Your Code with `FluxExecutor`

Now, your Jupyter environment is set up to use `flux-core`. You can run your code like this:

```python
import flux.job

# Use FluxExecutor within the Flux kernel
with flux.job.FluxExecutor() as flux_exe:
    print("FluxExecutor is running within the Jupyter Notebook")
```
