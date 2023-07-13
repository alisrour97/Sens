from setuptools import find_packages, setup

# with open("app/Readme.md", "r") as f:
#     long_description = f.read()

setup(
    name="sens",
    version="0.0.10",
    description="Closed Loop Sensitivity Framework",
    package_dir={"": "sens"},
    packages=find_packages(where="sens"),
    author="Ali SROUR",
    author_email="ali.srour@irisa.fr",
    license="MIT",
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10.6",
)