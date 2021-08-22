"""
trikit sample datasets. Includes Casualty Actuarial Society's Loss Reserving
Database.
The list of available datasets will continue to grow with each release.
Please check back periodically.
"""
from pathlib import Path

datasets_dir = Path(__file__).parent

dataref = {
    "raa": str(datasets_dir.joinpath("RAA.csv")),
    "ta83": str(datasets_dir.joinpath("TaylorAshe83.csv")),
    "lrdb": str(datasets_dir.joinpath("lrdb.csv")),
	"amw09": str(datasets_dir.joinpath("amw09.csv")),
    "autoliab": str(datasets_dir.joinpath("AutoLiabMedical.csv")),
    "glre": str(datasets_dir.joinpath("GLReinsurance2004.csv")),
    "singinjury": str(datasets_dir.joinpath("SingaporeInjury.csv")),
    "singproperty": str(datasets_dir.joinpath("SingaporeProperty.csv")),
    }
