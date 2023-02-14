#%%
from reco_model_mark2 import playerRecommend

import pandas as pd
import numpy as np

# %%
df = pd.read_csv(f"./data/NotNull.csv").drop(["Unnamed: 0"], axis=1)

emissionPlyer, recoList = playerRecommend(df=df, team='Chicago Bulls')

emissionPlyer
# %%
