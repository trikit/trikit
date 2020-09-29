"""
trikit sample datasets. Includes Casualty Actuarial Society's Loss Reserving
Database.
The list of available datasets will continue to grow with each release.
Please check back periodically.
"""
import os.path

datasets_dir = os.path.dirname(__file__)

dataref = {
    "nonstd": os.path.join(datasets_dir,  "nonstd.csv"),
    "raa": os.path.join(datasets_dir,  "RAA.csv"),
    "ta83": os.path.join(datasets_dir,  "TaylorAshe83.csv"),
    "lrdb": os.path.join(datasets_dir,  "lrdb.csv"),
    "autoliab": os.path.join(datasets_dir,  "AutoLiabMedical.csv"),
    "glre": os.path.join(datasets_dir,  "GLReinsurance2004.csv"),
    "singinjury": os.path.join(datasets_dir,  "SingaporeInjury.csv"),
    "singproperty": os.path.join(datasets_dir,  "SingaporeProperty.csv"),
    }

