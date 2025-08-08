from setuptools import setup, find_packages


setup(
    name="nano-eptk",
    version="0.0.1",
    packages=find_packages(
        where="utils",
    ),
    entry_points={
        "console_scripts": [
            "nano-eptk-mspoc = bin.mspoc:main",
            "nano-eptk-mspocr = bin.mspocr:main",
            "nano-eptk-poc = bin.poc:main",
            "nano-eptk-pocr = bin.pocr:main",
            "nano-eptk-saep = bin.saep:main",
            ]
        }
)
