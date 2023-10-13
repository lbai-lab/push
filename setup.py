import pathlib
from pathlib import Path
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


def get_install_requires() -> list[str]:
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    print("targets =========")
    print(targets)
    return targets

# This call to setup() does all the work
setup(
    name="PusH",
    version="0.1.0",
    description="Concurrent Probabilistic Programming for Bayesian Deep Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/lbai-lab/push/tree/main",
    author="The PusH Authors",
    author_email="dan.e.huang@gmail.com",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache 2.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_install_requires(),
    extras_require={
    }
)
