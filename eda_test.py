#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#%%
def corrMap(df, key=''):
    mask = np.zeros_like(df.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(30,20))
    plt.title(key)
    sns.heatmap(df.corr(), mask=mask, cmap='RdYlBu_r', linewidths=1, annot=True)
    plt.show()
    
# %%
plyDf = pd.read_csv("./data/NotNull.csv").drop(["Unnamed: 0", "pev"], axis=1)
teamDf = pd.read_csv("./data/teamObbs.csv").drop(["Unnamed: 0"], axis=1)

# %%
mdf = pd.merge(plyDf, teamDf, on=["season", "team"], how='inner')

# %%
corrMap(mdf)

# %%
