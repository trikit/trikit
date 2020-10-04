## `trikit` - A Pythonic Approach to Actuiarial Reserving

`trikit` is a collection of Loss Reserving utilities developed to facilitate
Actuarial analysis in Python, with particular emphasis on automating the basic
techniques generally used for estimating unpaid claim liabilities. 
trikit's core data structure is the triangle, which comes in both incremental
and cumulative varieties. trikit's triangle objects inherit directly
from Pandas DataFrame, so all of the familiar methods and attributes used
when working in Pandas can be be applied without modification to trikit's 
triangle objects. 

Along with the core `IncrTriangle` and `CumTriangle` data structures, 
trikit exposes a few common methods for estimating unpaid claim liabilities,
as well as techniques to quantify variability around those estimates. 
Currently available reserve estimators are `BaseChainLadder`, 
`MackChainLadder` and `BootstrapChainLadder`. Refer to the examples below 
for sample use cases. 



Finally, in addition to the library's core Chain Ladder functionality, `trikit`
exposes a convenient interface that links to the Casualty Actuarial Society's
Schedule P Loss Rerserving Database. The database contains information on
claims for major personal and commercial lines for all property-casualty
insurers that write business in the U.S[1]. For more information on
`trikit`'s Schedule P Loss Reserving Database API, check out the official
documentation [here](https://github.com/jtrive84/trikit/docs).
As of version 0.2.6, only the Commercial Auto Database is bundled with trikit.
Future releases will include additional loss reserving databases. 



## Installation

`trikit` can be installed by running:

```sh
$ python -m pip install trikit
```


## Quickstart Guide
---

We begin by loading the RAA dataset, which represents Automatic Factultative 
business in General Liability provided by the Reinsurance Association
of America. Sample datasets are loaded as DataFrame objects, and always 
represent incremental losses. Sample datasets can be loaded as follows:

```python
In [1]: import trikit
In [2]: raa = trikit.load("raa")
In [3]: raa.head()
Out[1]:
   origin  dev  value
0    1981    1   5012
1    1981    2   3257
2    1981    3   2638
3    1981    4    898
4    1981    5   1734
```

A list of available datasets can be obtained by calling `get_datasets`:

```python
In [4]: trikit.get_datasets()
Out[2]:
['raa', 'ta83', 'lrdb', 'autoliab', 'glre', 'singinjury', 'singproperty']
```

Any of the datasets listed above can be read in the same way using `trikit.load`. 
`trikit.load` takes additional  arguments to subset records
when accessing the CAS Loss Reserving Database. Refer to the docstring
for more information. 


### Working with Triangles

Triangles are created by calling the `totri` function. Available arguments
are:

- `data`: The dataset to transform into a triangle instance.   
<br>
- `type_`: {"cum", "incr"} Specifies the type of triangle to return.  
<br> 
- `data_format`: {"cum", "incr"} Specifies how losses are represented in `data`.  
<br> 
- `data_shape`: {"tabular", "triangle"} Specifies whether `data` represents
tabular loss data or data already structured as a loss triangle.  
<br>
- `origin`: The column name in `data` corresponding to accident year. 
Ignored if `data_shape="triangle"`.   
<br>
- `dev`: The column name in `data` corresponding to development period. 
Ignored if `data_shape="triangle"`.   
<br>
- `value`: The column name in `data` corresponding to the metric of interest. 
Ignored if `data_shape="triangle"`.    
<br>

Next we demonstrate how to create triangles using `totri` and various 
combinations of the arguments listed above.


#### Example \#1
*Creating a cumulative loss triangle from tabular incremental data.*
<br>

Referring again to the RAA dataset, let's create a cumulative loss triangle. 
We mentioned above that trikit sample datasets are Pandas DataFrames which 
reflect incremental losses, so `data_format="incr"` and `data_shape="tabular"`, 
both of which are defaults. Also, the default for `type_` is `"cum"`, so the 
only argument we need to pass into `totri` is the dataset:

```python
In [1]: import pandas as pd
In [2]: from trikit import load, totri
In [3]: raa = load("raa")
In [4]: tri = totri(raa)
In [5]: tri
Out[1]:
       1     2     3     4     5     6     7     8     9     10
1981 5012  8269 10907 11805 13539 16181 18009 18608 18662 18834
1982  106  4285  5396 10666 13782 15599 15496 16169 16704   nan
1983 3410  8992 13873 16141 18735 22214 22863 23466   nan   nan
1984 5655 11555 15766 21266 23425 26083 27067   nan   nan   nan
1985 1092  9565 15836 22169 25955 26180   nan   nan   nan   nan
1986 1513  6445 11702 12935 15852   nan   nan   nan   nan   nan
1987  557  4020 10946 12314   nan   nan   nan   nan   nan   nan
1988 1351  6947 13112   nan   nan   nan   nan   nan   nan   nan
1989 3133  5395   nan   nan   nan   nan   nan   nan   nan   nan
1990 2063   nan   nan   nan   nan   nan   nan   nan   nan   nan
```

`tri` is an instance of `trikit.triangle.CumTriangle`, which also inherits
from pandas.DataFrame:

```python
In [6]: type(tri)
Out[2]: trikit.triangle.CumTriangle
In [7]: isinstance(tri, pd.DataFrame)
Out[3]: True
```

This means that all of the really useful functionality made available by 
DataFrame objects can be applied to triangle objects. For example, to access
the first column of `tri`:

```python
In [8]: tri.loc[:,1]
Out[4]: 
1981   5012.00000
1982    106.00000
1983   3410.00000
1984   5655.00000
1985   1092.00000
1986   1513.00000
1987    557.00000
1988   1351.00000
1989   3133.00000
1990   2063.00000
Name: 1, dtype: float64
```

Triangle objects offer a number of methods useful in Actuarial reserving 
applications. To extract the latest diagonal, call `tri.latest`:

```python
In [9]: tri.latest
Out[5]:
   origin  dev      latest
0    1981   10 18834.00000
1    1982    9 16704.00000
2    1983    8 23466.00000
3    1984    7 27067.00000
4    1985    6 26180.00000
5    1986    5 15852.00000
6    1987    4 12314.00000
7    1988    3 13112.00000
8    1989    2  5395.00000
9    1990    1  2063.00000
```

Calling `tri.a2a` produces a DataFrame of age-to-age factors:

```python
In[10]: tri.a2a
Out[6]:
         1       2       3       4       5       6       7       8       9
1981  1.64984 1.31902 1.08233 1.14689 1.19514 1.11297 1.03326 1.00290 1.00922
1982 40.42453 1.25928 1.97665 1.29214 1.13184 0.99340 1.04343 1.03309     nan
1983  2.63695 1.54282 1.16348 1.16071 1.18570 1.02922 1.02637     nan     nan
1984  2.04332 1.36443 1.34885 1.10152 1.11347 1.03773     nan     nan     nan
1985  8.75916 1.65562 1.39991 1.17078 1.00867     nan     nan     nan     nan
1986  4.25975 1.81567 1.10537 1.22551     nan     nan     nan     nan     nan
1987  7.21724 2.72289 1.12498     nan     nan     nan     nan     nan     nan
1988  5.14212 1.88743     nan     nan     nan     nan     nan     nan     nan
1989  1.72199     nan     nan     nan     nan     nan     nan     nan     nan
```

Calling `tri.a2a_avgs`produces a table of candidate loss development factors, 
which contains arithmetic, geometric and weighted age-to-age averages for a 
number of different periods:

```python
In[11]: tri.a2a_avgs
Out[7]:
                   1       2       3       4       5       6       7       8       9
simple-1      1.72199 1.88743 1.12498 1.22551 1.00867 1.03773 1.02637 1.03309 1.00922
simple-2      3.43205 2.30516 1.11517 1.19815 1.06107 1.03347 1.03490 1.01799 1.00922
simple-3      4.69378 2.14200 1.21009 1.16594 1.10261 1.02011 1.03436 1.01799 1.00922
simple-4      4.58527 2.02040 1.24478 1.16463 1.10992 1.04333 1.03436 1.01799 1.00922
simple-5      5.42005 1.88921 1.22852 1.19013 1.12696 1.04333 1.03436 1.01799 1.00922
simple-6      4.85726 1.83148 1.35321 1.18293 1.12696 1.04333 1.03436 1.01799 1.00922
simple-7      4.54007 1.74973 1.31451 1.18293 1.12696 1.04333 1.03436 1.01799 1.00922
simple-8      9.02563 1.69589 1.31451 1.18293 1.12696 1.04333 1.03436 1.01799 1.00922
all-simple    8.20610 1.69589 1.31451 1.18293 1.12696 1.04333 1.03436 1.01799 1.00922
geometric-1   1.72199 1.88743 1.12498 1.22551 1.00867 1.03773 1.02637 1.03309 1.00922
geometric-2   2.97568 2.26699 1.11513 1.19783 1.05977 1.03346 1.03487 1.01788 1.00922
geometric-3   3.99805 2.10529 1.20296 1.16483 1.10019 1.01993 1.03433 1.01788 1.00922
geometric-4   4.06193 1.98255 1.23788 1.16380 1.10802 1.04244 1.03433 1.01788 1.00922
geometric-5   4.73672 1.83980 1.22263 1.18840 1.12492 1.04244 1.03433 1.01788 1.00922
geometric-6   4.11738 1.78660 1.32455 1.18138 1.12492 1.04244 1.03433 1.01788 1.00922
geometric-7   3.86345 1.69952 1.28688 1.18138 1.12492 1.04244 1.03433 1.01788 1.00922
geometric-8   5.18125 1.64652 1.28688 1.18138 1.12492 1.04244 1.03433 1.01788 1.00922
all-geometric 4.56261 1.64652 1.28688 1.18138 1.12492 1.04244 1.03433 1.01788 1.00922
weighted-1    1.72199 1.88743 1.12498 1.22551 1.00867 1.03773 1.02637 1.03309 1.00922
weighted-2    2.75245 2.19367 1.11484 1.19095 1.05838 1.03381 1.03326 1.01694 1.00922
weighted-3    3.24578 2.05376 1.23215 1.15721 1.09340 1.02395 1.03326 1.01694 1.00922
weighted-4    3.47986 1.91259 1.26606 1.15799 1.09987 1.04193 1.03326 1.01694 1.00922
weighted-5    4.23385 1.74821 1.24517 1.17519 1.11338 1.04193 1.03326 1.01694 1.00922
weighted-6    3.30253 1.70935 1.29886 1.17167 1.11338 1.04193 1.03326 1.01694 1.00922
weighted-7    3.16672 1.67212 1.27089 1.17167 1.11338 1.04193 1.03326 1.01694 1.00922
weighted-8    3.40156 1.62352 1.27089 1.17167 1.11338 1.04193 1.03326 1.01694 1.00922
all-weighted  2.99936 1.62352 1.27089 1.17167 1.11338 1.04193 1.03326 1.01694 1.00922
```

We can obtain a reference to an incremental version of `tri` by calling
`to_incr`:

```python
In[12]: tri.to_incr()
Out[8]:
      1    2    3    4    5    6    7   8   9   10
1981 5012 3257 2638  898 1734 2642 1828 599  54 172
1982  106 4179 1111 5270 3116 1817 -103 673 535 nan
1983 3410 5582 4881 2268 2594 3479  649 603 nan nan
1984 5655 5900 4211 5500 2159 2658  984 nan nan nan
1985 1092 8473 6271 6333 3786  225  nan nan nan nan
1986 1513 4932 5257 1233 2917  nan  nan nan nan nan
1987  557 3463 6926 1368  nan  nan  nan nan nan nan
1988 1351 5596 6165  nan  nan  nan  nan nan nan nan
1989 3133 2262  nan  nan  nan  nan  nan nan nan nan
1990 2063  nan  nan  nan  nan  nan  nan nan nan nan
```


#### Example \#2
*Create an incremental loss triangle from tabular incremental data.*  
<br>
The call to `totri` is identical to Example #1, but we change `type_` from 
"cum" to "incr":

```python
In [1]: import pandas as pd
In [2]: from trikit import load, totri
In [3]: raa = load("raa")
In [4]: tri = totri(raa, type_="incr")
In [5]: type(tri)
Out[1]: trikit.triangle.IncrTriangle
In [6]: tri
      1    2    3    4    5    6    7   8   9   10
1981 5012 3257 2638  898 1734 2642 1828 599  54 172
1982  106 4179 1111 5270 3116 1817 -103 673 535 nan
1983 3410 5582 4881 2268 2594 3479  649 603 nan nan
1984 5655 5900 4211 5500 2159 2658  984 nan nan nan
1985 1092 8473 6271 6333 3786  225  nan nan nan nan
1986 1513 4932 5257 1233 2917  nan  nan nan nan nan
1987  557 3463 6926 1368  nan  nan  nan nan nan nan
1988 1351 5596 6165  nan  nan  nan  nan nan nan nan
1989 3133 2262  nan  nan  nan  nan  nan nan nan nan
1990 2063  nan  nan  nan  nan  nan  nan nan nan nan
```

`tri` now represents RAA losses in incremental format.  
<br>
It is possible to obtain a cumulative representation of an incremental triangle
object by calling `tri.to_cum`:

```python
In [7]: tri.to_cum()
  1     2     3     4     5     6     7     8     9     10
1981 5012  8269 10907 11805 13539 16181 18009 18608 18662 18834
1982  106  4285  5396 10666 13782 15599 15496 16169 16704   nan
1983 3410  8992 13873 16141 18735 22214 22863 23466   nan   nan
1984 5655 11555 15766 21266 23425 26083 27067   nan   nan   nan
1985 1092  9565 15836 22169 25955 26180   nan   nan   nan   nan
1986 1513  6445 11702 12935 15852   nan   nan   nan   nan   nan
1987  557  4020 10946 12314   nan   nan   nan   nan   nan   nan
1988 1351  6947 13112   nan   nan   nan   nan   nan   nan   nan
1989 3133  5395   nan   nan   nan   nan   nan   nan   nan   nan
1990 2063   nan   nan   nan   nan   nan   nan   nan   nan   nan
```


#### Example \#3
*Create an cumulative loss triangle from data formatted as a triangle*.  
<br>
There may be situations in which data is already formatted as a triangle, 
and we're interested in creating a triangle instance from this data. 
In the next example, we create a DataFrame with the same shape as a triangle, 
which we then pass into `totri` with `data_shape="triangle"` to obtain a 
cumulative triangle instance:

```python
In [1]: import pandas as pd
In [2]: from trikit import load, totri
In [3]: dftri = pd.DataFrame({
            1:[1010, 1207, 1555, 1313, 1905],
            2:[767, 1100, 1203, 900, np.NaN],
            3:[444, 623, 841, np.NaN, np.NaN],
            4:[239, 556, np.NaN, np.NaN, np.NaN],
            5:[80, np.NaN, np.NaN, np.NaN, np.NaN],
            }, index=list(range(1, 6))
            )
In [4]: dftri
Out[1]:
    1          2         3         4        5
1  1010  767.00000 444.00000 239.00000 80.00000
2  1207 1100.00000 623.00000 556.00000      nan
3  1555 1203.00000 841.00000       nan      nan
4  1313  900.00000       nan       nan      nan
5  1905        nan       nan       nan      nan

In [5]: tri = totri(dftri, data_shape="triangle")
In [6]: type(tri)
Out[2]: trikit.triangle.CumTriangle 
```

trikit cumulative triangle instances expose a plot method, which generates a 
faceted plot by origin representing the progression of cumulative losses to 
date by development period. The exhibit can be obtained as follows:

```python
In [5]: tri.plot()
```

## Chain Ladder Estimates

In trikit, chain ladder reserve estimates are obtained by calling a cumulative
triangle's `cl` method. Let's refer to the CAS Loss Reserving Dastabase 
included with trikit, focusing `grcode=1767` (`grcode` uniquely identifies 
each company in the database. To obtain a full list of grcodes and their
corresponding companies, use `trikit.get_lrdb_groups()`):

```python
In [1]: from trikit import load, totri
In [2]: df = load("lrdb", grcode=1767)
In [3]: tri = totri(df)
In [4]: result = tri.cl()
In [5]: result
   origin maturity     cldf   latest ultimate  reserve
0    1988       10  1.00000  1752096  1752096        0
1    1989        9  1.12451  1633619  1837022   203403
2    1990        8  1.28233  1610193  2064802   454609
3    1991        7  1.49111  1278228  1905977   627749
4    1992        6  1.77936  1101390  1959771   858381
5    1993        5  2.20146   980180  2157822  1177642
6    1994        4  2.87017   792392  2274299  1481907
7    1995        3  4.07052   560278  2280624  1720346
8    1996        2  6.68757   326584  2184053  1857469
9    1997        1 15.62506   143970  2249541  2105571
10  total               nan 10178930 20666007 10487077
```

result is of type `chainladder.BaseChainLadderResult`.    

When the `range_method` argument of `cl` is None, two keyword arguments
can be provided:

- `tail`: The tail factor, which defaults to 1.0 
- `sel`: Loss development factors, which defaults to "all-weighted".

Recall from Example #2 we demonstrated how to access a number of candidate loss 
development patterns by calling `tri.a2a_avgs`. Available options for `sel` can 
be any value present in `tri.a2a_avgs`'s index. To obtain a list of available 
loss development factors by name, run:

```python
In [1]: tri.a2a_avgs.index.tolist()
Out[1]:
['simple-1', 'simple-2', 'simple-3', 'simple-4', 'simple-5', 'simple-6', 'simple-7', 
 'simple-8', 'all-simple', 'geometric-1', 'geometric-2', 'geometric-3', 'geometric-4', 
 'geometric-5', 'geometric-6', 'geometric-7', 'geometric-8', 'all-geometric', 
 'weighted-1', 'weighted-2', 'weighted-3', 'weighted-4', 'weighted-5', 'weighted-6', 
 'weighted-7', 'weighted-8', 'all-weighted']
```

If instead of `all-weighted`, a 5-year geometric loss 
development pattern is preferred, along with a tail factor of 1.015, 
the call to `cl` becomes:

```python
In [1]: tri.cl(sel="geometric-5", tail=1.015)
Out[1]:
   origin maturity     cldf   latest ultimate  reserve
0    1988       10  1.01500  1752096  1778377    26281
1    1989        9  1.14138  1633619  1864578   230959
2    1990        8  1.30157  1610193  2095778   485585
3    1991        7  1.51344  1278228  1934517   656289
4    1992        6  1.80591  1101390  1989009   887619
5    1993        5  2.23416   980180  2189878  1209698
6    1994        4  2.91249   792392  2307832  1515440
7    1995        3  4.13521   560278  2316869  1756591
8    1996        2  6.78292   326584  2215194  1888610
9    1997        1 15.69149   143970  2259103  2115133
10  total               nan 10178930 20951135 10772205
```

A faceted plot by origin comparing combining actuals and estimates can 
be obtained by calling the `BaseChainLadderResult`'s plot method:

```python
In [1]: result = tri.cl(sel="geometric-5", tail=1.015)
In [2]: result.plot()
```


### Quantifying Reserve Variability

The base chain ladder method provides an estimate by origin and in total of 
future claim liabilities, but offers no indication of the variability around 
those point estimates. We can obtain quantiles of the predictive distribution 
of reserve estimates by setting `range_method="bootstrap"`. When `range_method`
is set to "bootstrap", available optional parameters include:

- `sims`: The number of bootstrap iterations to perform. Default value is 1000.   
<br>      
- `q`: Quantile or sequence of quantiles to compute, which must be between 0 
and 1 inclusive. Default value is [.75, .95].  
<br>   
- `neg_handler`: Determines how negative incremental triangle values should be 
handled. If set to "first", cells with value less than 0 will be set to 1. If 
set to "all", the minimum value in all triangle cells is identified ('MIN_CELL'). 
If MIN_CELL is less than or equal to 0, `MIN_CELL + X = +1.0` is solved for `X`. 
`X` is then added to every other cell in the triangle, resulting in all 
incremental triangle cells having a value strictly greater than 0. Default
value is first.     
<br>
`procdist`: The distribution used to incorporate process variance. Currently,
this can only be set to "gamma". This may change in a future release.  
<br>
`two_sided`: Whether the two_sided prediction interval should be included in 
summary output. For example, if ``two_sided=True`` and ``q=.95``, then
the 2.5th and 97.5th quantiles of the predictive reserve distribution will be 
returned [(1 - .95) / 2, (1 + .95) / 2]. When False, only the specified 
quantile(s) will be included in summary output. Default value is False.   
<br>
`parametric`:  If True, fit standardized residuals to a normal distribution via
maximum likelihood, and sample from this parameterized distribution. Otherwise, 
sample with replacement from the collection of standardized fitted triangle 
residuals. Default value to False.     
<br>
`interpolation`: One of {'linear', 'lower', 'higher', 'midpoint', 'nearest'}.
Default value is "linear". Refer to [`numpy.quantile`](https://numpy.org/devdocs/reference/generated/numpy.quantile.html) for more information.  
<br>
`random_state`:  If int, random_state is the seed used by the random number
generator; If `RandomState` instance, random_state is the random number generator; 
If None, the random number generator is the `RandomState` instance used by 
np.random. Default value is None.     
<br>

EThe suggested approach is to collect parameters into a dictionary, 
then include the dictionary with the call to the triangle's `cl` method. 
We next demonstrate how to apply the bootstrap chainladder to the raa dataset.
We'll set `sims=2500`, `two_sided=True` and `random_state=516`:

```python
In [1]: from trikit import load, totri
In [2]: df = load("raa")
In [3]: tri = totri(data=df)
In [4]: bclargs = {"sims":2500, "two_sided":True, "random_state":516}
In [5]: bcl = tri.cl(range_method="bootstrap", **bclargs)
In [6]: bcl
Out[1]:
   origin maturity    cldf latest ultimate  cl_reserve  bcl_reserve  2.5% 12.5% 87.5%  97.5%
0    1981       10 1.00000  18834    18834     0.00000      0.00000     0     0     0      0
1    1982        9 1.00922  16704    16858   153.95392      4.94385  -691   -71   543   1610
2    1983        8 1.02631  23466    24083   617.37092    404.09648 -1028  -100  1727   3115
3    1984        7 1.06045  27067    28703  1636.14216   1377.04868  -518   227  3351   5129
4    1985        6 1.10492  26180    28927  2746.73634   2423.95365    50   859  4826   7209
5    1986        5 1.23020  15852    19501  3649.10318   3457.84768   724  1688  5986   8226
6    1987        4 1.44139  12314    17749  5435.30259   5289.49722  1536  2730  8622  11521
7    1988        3 1.83185  13112    24019 10907.19251  10635.06275  4477  6577 15557  20131
8    1989        2 2.97405   5395    16045 10649.98410  10247.20301  2824  5452 16603  21204
9    1990        1 8.92023   2063    18402 16339.44253  15480.77315   565  5164 29130  41923
10  total              nan 160987   213122 52135.22826  49320.42648  7938 22526 86344 120069
```

Here `cl_reserve` represents standard chainladder reserve point estimates. 
`bcl_reserve` represents the 50th percentile of the predicitive distribution 
of reserve estimates by origin and in total, and `2.5%`, `12.5%`, `87.5%` and `97.5%`
represent various percentiles of the predictive distribution of reserve estimates. 
The lower percentiles,  `2.5%` and `12.5%` are included because `two_sided=True`. 
If `two_sided=False`, they would not be included, and the included percentiles 
would be `75%` and `95%`.

## Looking Ahead
In future releases, trikit will include additional methods to quantify reserve 
variability, including the Mack method and various Markov Chain Monte Carlo 
approaches.   
<br>
Please contact james.triveri@gmail.com with suggestions or feature requests.



## Relevant Links

- [trikit Source](https://github.com/trikit/trikit)
- [CAS Loss Reserving Database](https://www.casact.org/research/index.cfm?fa=loss_reserves_data)
- [Python](https://www.python.org/)
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
