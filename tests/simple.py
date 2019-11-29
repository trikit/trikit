import sys; sys.path.append("G:\\Repos")
import trikit
from trikit.chainladder import BaseChainLadder




raa  = trikit.load(dataset="raa")
ta83 = trikit.load(dataset="ta83")
auto = trikit.load(dataset="lrdb", lob="COM_AUTO", grcode=32743)



tri = trikit.totri(data=raa)
cl = BaseChainLadder(tri)

