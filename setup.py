import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="PusH",
    version="0.1.0",
    description="Concurrent Probabilistic Programming for Bayesian Deep Learning",
    long_description=README,
    long_description_content_type="text/markdown",
    url="coming soon...",
    author="The PusH Authors",
    author_email="",
    license="Apache 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache 2.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        ],
    extras_require={
    }
)
