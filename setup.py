import yaml

from setuptools import setup, find_packages
from os import path

# read the contents of README file
this_dir = path.abspath(path.dirname(__file__))

with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="neurogenia",
    version=yaml.load(open("changelog.yml"))['versions'][-1]['name'],
    packages=["neurogenia"],
    
    # metadata to display on PyPI
    author="Gustavo6046",
    author_email="gugurehermann@gmail.com",
    description="A neural network and machine learning library.",
    license="MIT",
    keywords="neural machine_learning neural_network genetic genetic_algorithm",
    url="https://github.com/Gustavo6046/Neurogenia",
    
    long_description=long_description,
    long_description_content_type='text/markdown'
)