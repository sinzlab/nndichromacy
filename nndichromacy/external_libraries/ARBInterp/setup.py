from distutils.core import setup

setup(
    name="ARBInterp",
    version="1.6",
    packages=[
        "ARBTools",
    ],
    author=["Paul Walker"],
    author_email="paul.a.walker@durham.ac.uk",
    url="https://www.jqc.org.uk/",
    description="Python tools for interpolating 3D or 4D fields",
    long_description=open("README.txt", "r").read(),
    license="GPL 3",
)
