import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

DATA_FILES = ['img/new_loss.png']


install_requires = [
    'numpy',
    'scikit-learn',
    'tqdm',
    #'torch',
    #'torchvision',
    ],



setuptools.setup(
    name="ptranking",
    version="0.3",
    author="II-Research",
    author_email="example@example.com",
    description="A library of scalable and extendable implementations of typical learning-to-rank methods based on PyTorch.",
    license="MIT License",
    keywords=['Learning-to-rank', 'PyTorch'],
    url="https://ptranking.github.io",
    packages=setuptools.find_namespace_packages(include=["ptranking", "ptranking.*"]),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = install_requires,
    data_files = DATA_FILES
)
