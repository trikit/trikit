## `trikit` - Actuarial Reserving Methods in Python

`trikit` is a collection of Loss Reserving utilities developed to facilitate
Actuarial analysis in Python, with particular emphasis on automating the basic
techniques generally used for estimating unpaid claim liabilities. `trikit`
currently implements the Chain Ladder method for ultimate loss projection,
along with routines to compute the Chain Ladder prediction error, which can be
used to quantify the variability around the ultimate loss projection
point estimates.

In addition to the library's core Chain Ladder functionality, `trikit`
exposes a convenient interface that links to the Casualty Actuarial Society's
Schedule P Loss Rerserving Database. The database contains information on
claims for major personal and commercial lines for all property-casualty
insurers that write business in the U.S[1]. For more information on
`trikit`'s Schedule P Loss Reserving Database API, check out the official
documentation [here](https://github.com/jtrive84/trikit/docs).


## Installation

`trikit` can be installed by running:

```sh
$ pip install trikit (NOTE: Not yet on PyPI - Coming soon)
```

Alternatively, manual installation can be accomplished by downloading the
source archive, extracting the contents and running:

```sh
$ python setup.py install
```


## Relevant Links
- [trikit Quickstart Guide](https://github.com/jtrive84/trikit/docs/quickstart)
- [trikit Documentation](https://github.com/jtrive84/trikit/docs)
- [trikit Source](https://github.com/jtrive84/trikit)
- [CAS Loss Reserving Database](https://www.casact.org/research/index.cfm?fa=loss_reserves_data)
- [Python](https://www.python.org/)
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)





### Footnotes     

[1] https://www.casact.org/research/index.cfm?fa=loss_reserves_data
