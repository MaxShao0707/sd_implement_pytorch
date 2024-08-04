from setuptools import setup, find_packages

#read readme file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stable_diffusion_pytorch",
    version="0.1.0",
    author="max shaw",
    author_email="ml20xs@leeds.ac.uk",
    description="implementation stable diffusion by using pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaxShao0707/sd_implement_pytorch.git",
    packages=find_packages(),

) 