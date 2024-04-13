import setuptools

DISTNAME = "sparsepoly"
DESCRIPTION = (
    "Sparse factorization machines and their variants"
    "for classification and regression in Python."
)
LONG_DESCRIPTION = open("README.md").read()
MAINTAINER = "Kyohei Atarashi"
MAINTAINER_EMAIL = "atarashi@i.kyoto-u.ac.jp"
LICENSE = "MIT"
VERSION = "0.4.dev0"


if __name__ == "__main__":
    setuptools.setup(
        name=DISTNAME,
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],
    )
