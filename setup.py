from setuptools import setup, find_packages

setup(
    name="sirentv",
    version="0.0.1",
    packages=find_packages(),
    # photonlib and slar (siren-lartpc) are not on PyPI: install them from their repositories
    install_requires=[
        "torch",
        "numpy",
        "h5py",
        "pyyaml",
        "tqdm",
        "wandb",
        "matplotlib",  # training-time wandb plots
        "scipy",       # eval power spectra (inpainting)
    ],
    entry_points={
        "console_scripts": [
            "sirentv-train=sirentv.train:main",
            "sirentv-eval=sirentv.eval.eval:main",
        ],
    },
    author="Sam Young",
    author_email="youngsam@stanford.edu",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CIDeR-ML/sirentv",
    python_requires=">=3.10",  # PEP 604 annotations evaluated at runtime
)
