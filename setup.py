from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    distname="accord-nlp",
    version="1.0.2",
    author="Hansi Hettiarachchi",
    author_email="hansi.h.hettiarachchi@gmail.com",
    description="ACCORD-NLP: Transformer/language model-based information extraction from regulatory text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Accord-Project/accord-nlp",
    keywords=['NLP', 'NER', 'Relation Extraction', 'Information Extraction'],
    packages=find_packages(exclude=("data", "data_annotation", "experiments", "data_preparation", "data_processing",
                                    "rase-to-json", "sentence-classification", "text-to-rase")),
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
        "transformers==4.40.2",
        "seqeval",
        "tensorboardX",
        "wandb",
        "sentencepiece",
        "datasets",
        "graphviz"
    ],
)
