from setuptools import setup


setup(
    name="nano-eptk",
    version="0.0.1",
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
