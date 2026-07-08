from setuptools import setup


setup(
    name="nano-eptk",
    entry_points={
        "console_scripts": [
            "mspoc = nano_eptk.bin.mspoc:main",
            "mspocr = nano_eptk.bin.mspocr:main",
            "poc = nano_eptk.bin.poc:main",
            "pocr = nano_eptk.bin.pocr:main",
            "saep = nano_eptk.bin.saep:main",
            "dhcp_gre_saep = nano_eptk.bin.dhcp_gre_saep:main",
            "dhcp_epi_poc = nano_eptk.bin.dhcp_epi_poc:main",
            "dhcp_epi_pocr = nano_eptk.bin.dhcp_epi_pocr:main",
            "dhcp_tse_mspoc = nano_eptk.bin.dhcp_tse_mspoc:main",
            "dhcp_tse_mspocr = nano_eptk.bin.dhcp_tse_mspocr:main",
            "nano-eptk = nano_eptk.bin.dispatch:main",
            ]
        }
)
