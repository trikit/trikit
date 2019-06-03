import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trikit",
    version="0.2.4",
    author="James D. Triveri",
    author_email="james.triveri@gmail.com",
    description="Actuarial Reserving Methods in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trikit/trikit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    install_requires=[
        "numpy>=1.*", "scipy>=0.19", "pandas>=0.20", "matplotlib>=2.*",
        "seaborn>=0.7",
        ],
    include_package_data=True,
    )
