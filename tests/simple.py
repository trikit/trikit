import sys; sys.path.append("G:\\Repos")
import trikit
from trikit.chainladder import BaseChainLadder





raa  = trikit.load(dataset="raa")
ta83 = trikit.load(dataset="ta83")
auto = trikit.load(dataset="lrdb", loss_key="COM_AUTO", grcode=32743)
nstd = trikit.load(dataset="nonstd")



tri = trikit.totri(data=raa)

