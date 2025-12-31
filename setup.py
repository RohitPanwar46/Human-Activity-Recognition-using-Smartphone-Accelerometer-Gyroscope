from setuptools import setup, find_packages

setup(
    name="har",                 # package name (import har)
    version="0.1.0",
    package_dir={"": "src"},     # telling Python: code is inside src/
    packages=find_packages(where="src"),
)
