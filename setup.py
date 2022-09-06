"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
from pathlib import Path
import versioneer


setup(
    name='pympipool',
    version=versioneer.get_version(),
    description='pympipool - scale functions over multiple compute nodes using mpi4py',
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
        'Programming Language :: Python :: 3.10'
    ],
    packages=find_packages(exclude=["*tests*", "*.ci_support*"]),
    install_requires=[
        'dill==0.3.5.1',
        'mpi4py==3.1.3',
        'tqdm==4.64.1'
    ],
    cmdclass=versioneer.get_cmdclass(),

    entry_points={
            "console_scripts": [
                'pympipool=pympipool.__main__:main'
            ]
    }
)
