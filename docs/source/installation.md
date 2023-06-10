# Installation
The `pympipool` package can be installed either via `pip` or `conda`. While most HPC systems use Linux these days, the `pympipool` package can be installed on all major operation systems. 

## pypi-based installation
`pympipool` can be installed from the python package index (pypi) using the following command: 
```
pip install pympipool
```

## conda-based installation 
The `conda` package combines all dependencies in one package: 
```
conda install -c conda-forge pympipool
```
When resolving the dependencies with `conda` gets slow it is recommended to use `mamba` instead of `conda`. So you can also install `pympipool` using: 
```
mamba install -c conda-forge pympipool
```
