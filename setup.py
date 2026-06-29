from setuptools import setup


setup(
    name="nano-eptk",
    entry_points={
        "console_scripts": [
            "mspoc = bin.mspoc:main",
            "mspocr = bin.mspocr:main",
            "poc = bin.poc:main",
            "pocr = bin.pocr:main",
            "saep = bin.saep:main",
            "dhcp_gre_saep = bin.dhcp_gre_saep:main",
            "dhp_epi_poc = bin.dhp_epi_poc:main",
            "dhp_epi_pocr = bin.dhp_epi_pocr:main",
            "dhcp_tse_mspoc = bin.dhcp_tse_mspoc:main",
            "dhcp_tse_mspocr = bin.dhcp_tse_mspocr:main",
            ]
        }
)
