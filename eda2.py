#%%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv(r"./data/player_stats.csv").drop(["#", "Unnamed: 0"], axis=1)
df

# %%
df.columns = df.columns.str.replace("\r\n", "_")
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("__", "_")
df.columns.tolist()

# %%
df.isna().sum().tolist()

# %%
df

# %%
