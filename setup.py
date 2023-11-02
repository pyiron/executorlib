"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
from pathlib import Path
import versioneer


setup(
    name='pympipool',
    version=versioneer.get_version(),
    description='pympipool - Scale serial and MPI-parallel python functions over hundreds of compute nodes all from within a jupyter notebook or serial python process.',
    long_description=Path("README.md").read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/jan-janssen/pympipool',
    author_email='jan.janssen@outlook.com',
    license='BSD',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    packages=find_packages(exclude=["*tests*", "*.ci_support*"]),
    install_requires=[
        'cloudpickle==3.0.0',
        'mpi4py==3.1.5',
        'tqdm==4.66.1',
        'pyzmq==25.1.1',
    ],
    cmdclass=versioneer.get_cmdclass(),
)
