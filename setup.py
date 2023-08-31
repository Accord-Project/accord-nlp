from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    distname="accord-nlp",
    version="0.1.8",
    author="Hansi Hettiarachchi",
    author_email="hansi.h.hettiarachchi@gmail.com",
    description="ACCORD - Information extraction from text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Accord-Project/NLP-Framework",
    keywords=['NLP', 'NER', 'Relation Extraction'],
    packages=find_packages(exclude=("data_annotation", "experiments", "data_preparation")),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "openpyxl",
        "nltk",
        "transformers==4.16.2",
        "seqeval",
        "tensorboardX",
        "wandb",
        "sentencepiece",
        "graphviz"
    ],
)
