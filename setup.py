from setuptools import setup


setup(
    name="nano-eptk",
    entry_points={
        "console_scripts": [
            "mspoc = nano_eptk.cli.mspoc:main",
            "mspocr = nano_eptk.cli.mspocr:main",
            "poc = nano_eptk.cli.poc:main",
            "pocr = nano_eptk.cli.pocr:main",
            "saep = nano_eptk.cli.saep:main",
            "dhcp_gre_saep = nano_eptk.cli.dhcp_gre_saep:main",
            "dhcp_epi_poc = nano_eptk.cli.dhcp_epi_poc:main",
            "dhcp_epi_pocr = nano_eptk.cli.dhcp_epi_pocr:main",
            "dhcp_tse_mspoc = nano_eptk.cli.dhcp_tse_mspoc:main",
            "dhcp_tse_mspocr = nano_eptk.cli.dhcp_tse_mspocr:main",
            "nano-eptk = nano_eptk.cli.dispatch:main",
            ]
        }
)
