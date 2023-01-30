#%%
import pandas as pd
import numpy as np

# %%
newDf = pd.read_csv(f"./data/new_df.csv")
# %%
positions = newDf.position.unique()

# %%
for pos in positions:
    df = newDf[newDf['position']==pos]
    print("\n\n")
    print(df["new_value"].describe())
    
# %%
