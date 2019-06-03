import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trikit",
    version="0.2.2",
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
    include_package_data=True,
    )
