import sys; sys.path.append("C:\\Users\\cac9159\\Repos\\")
import trikit
from trikit import totri, chladder

# Load sample dataset.
raa = trikit.load(dataset="raa")
DATA = raa


# Convert dataset with origin, dev and value to a triangle object:
tri = trikit.totri(DATA, type_="cum")


# Bind reference to latest_diagonal:
latest = tri.latest


# Generate age-to-age factors:
a2a = tri.a2a


# Review all computed ldf averages:
avgs = tri.a2a_avgs


# Visualize triangle development to date as lattice plot:
tri.plot()


# Generate incremental triangle:
itri = tri.as_incr()



res_ = tri.cl(sel="all-weighted", tail=1.0, range_method=None)






# Chain Ladder demonstration =================================================]

# Pass original dataset into top-level chladder function:
cl = chladder(data=DATA, range_method=None)


# In order to produce Chain Ladder reserve estimates, call cl's run method:
clsumm = cl.run(sel="all-weighted", tail=1.0)


# Produce Chain Ladder reserve estimate summary:
clsumm


# View squared traingle:
clsumm.trisqrd.round()

# View projected development along with actual development to date:
clsumm.plot()

# Deomonstration of trikit's range generation functionality ==================]

# Pass original dataset into top-level chladder function, but specify
# "bootstrap" for range_method:
bcl = chladder(data=DATA, range_method="bootstrap")

bclsumm = bcl.run(sims=1000)

