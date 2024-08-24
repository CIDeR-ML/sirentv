from setuptools import setup, find_packages

setup(
    name="sirentv",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
        "tqdm",
        "wandb",
    ],
    entry_points={
        "console_scripts": [
            "sirentv-train=sirentv.train:train",
        ],
    },
    author="Sam Young",
    author_email="youngsam@stanford.edu",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/youngsm/sirentv",
    python_requires='>=3.7',
    include_package_data=True,
    package_data={
        'sirentv': ['templates/*.yaml', 'data/*.npz'],
    },
)
