# Delete build
# Delete dist
# Delete trikit.egg-info

# Create .pypirc file in $HOME:
[distutils]
index-servers=
    pypi
    testpypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: jtrive84
password: Principia1687

[pypi]
username: jtrive84
password: Principia1687

# Run:
$ python setup.py sdist bdist_wheel --universal
$ twine upload dist/*