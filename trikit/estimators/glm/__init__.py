"""
Generalized Linear Model reserve estimators.
"""
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf

class GLM:
	
	def __init__(self, tri):
		
		raise NotImplementedError("Not yet implemented.")
		# df = trikit.load("ta83")
		#~ df["dev"] = df["dev"] * 12
		#~ df["origin"] = df["origin"] + 2000

		#~ tri = trikit.totri(df, tri_type="incr")
		#~ dfpred = tri.to_tbl(dropna=False).rename({"value":"y"}, axis=1)
		#~ dfpred["pred_ind"] = dfpred["y"].map(lambda v: 1 if np.isnan(v) else 0)

		#~ mdl = smf.glm(
			#~ formula="value ~ C(origin) + C(dev)", data=df,
			#~ family=sm.families.Gamma(link=sm.families.links.log())
			#~ ).fit()

		#~ dfpred["yhat"] = mdl.predict(dfpred)
		#~ dfpred["yhat"] = dfpred.apply(lambda rec: rec.y if rec.pred_ind==0 else rec.yhat, axis=1)
		#~ dfpred = dfpred[["origin", "dev", "yhat"]].rename({"yhat":"value"}, axis=1)

		#~ trisqrd = trikit.totri(dfpred, tri_type="cum", data_format="incr")






class GLMResult:
	
	def __init__(self, **kwargs):
		
		raise NotImplementedError("Not yet implemented.")
		



