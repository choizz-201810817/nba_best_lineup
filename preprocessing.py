#%%
import pandas as pd
import numpy as np
# %%
teamDf = pd.read_csv(f"./data/team_stats.csv")
# %%
teamDf = teamDf.drop(['Unnamed: 0', '#'], axis=1)
teamDf

# %%
teamDf.season = teamDf['season'].apply(lambda x: '20'+x[-2:])

# %%
teamDf['teamId'] = teamDf.team + '-' + teamDf.season

# %%
teamDf.teamId
# %%
teamDf.to_csv(f"./data/team_data.csv")

# %%
plyDf = pd.read_csv(f"./data/player_stats.csv")
# %%
plyDf = plyDf.drop(['Unnamed: 0', '#'], axis=1)
plyDf
# %%
plyDf.season = plyDf['season'].apply(lambda x: '20'+x[-2:])

#%%
plyDf['teamId'] = plyDf.team + '-' + plyDf.season
plyDf
# %%
plyDf['playerId'] = plyDf.player + '-' + plyDf.season
# %%
plyDf.to_csv('./data/player_data.csv')

# %%
df = pd.read_csv(f"./nonCorr.csv").drop(["Unnamed: 0"], axis=1)

# %%
df1 = df[df['position']=='C']
df1['new'] =  (2.05*df1['netrtg']) + (1.80*df1['pitp']) + (1.27*df1['def_ws']) + (1.17*df1['poss']) + (0.57*df1['defrtg']) + (0.55*df1['%pts']) + (0.37*df1['dreb%']) + (0.30*df1['%pts_fbps']) + (0.06*df1['obbs']) + (0.04*df1['pie']) + (0.03*df1['oreb%']) - (2.37*df1['age']) - (0.81*df1['%pf']) - (0.47*df1['pf'])

# %%
df1.new.describe()

# %%
