"""
trikit sample datasets. Includes Casualty Actuarial Society's Loss Reserving
Database.
The list of available datasets will continue to grow with each release.
Please check back periodically.
"""
import os.path

datasets_dir = os.path.dirname(__file__)

dataref = {
    "nonstd": datasets_dir + os.path.sep + "nonstd.csv",
    "raa"   : datasets_dir + os.path.sep + "RAA.csv",
    "ta83"  : datasets_dir + os.path.sep + "TaylorAshe83.csv",
    "lrdb"  : datasets_dir + os.path.sep + "lrdb.csv",
    }

