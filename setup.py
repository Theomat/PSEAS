from setuptools import setup, find_packages


setup(
    name="pseas",
    version="0.0.1",
    description="Per Set Efficient Algorithm Selection (PSEAS)",
    author="Théo Matricon",
    author_email="theomatricon@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pyyaml",
        "liac-arff",
        "pandas",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
    license="?",
)
