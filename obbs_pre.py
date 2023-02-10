#%%
import pandas as pd
import numpy as np

# %%
teamDf = pd.read_csv("./data/team_stats.csv")
teamDf['obbs'] = teamDf.w / teamDf.gp
teamDf.season = teamDf.season.apply(lambda x: '20'+x[-2:])
teamDf = teamDf[["team","season","obbs"]]
teamDf

# %%
plyDf = pd.read_csv("./data/nonCorrAllPos.csv").drop(["Unnamed: 0", "inflation_salary"], axis=1)
plyDf

# %%
plyDf.groupby(["season","team","position"]).mean()

# %%
def changedTeam(x=''):
    if x=='New Jersey Nets':
        return 'Brooklyn Nets'
    elif x=='New Orleans Hornets':
        return 'New Orleans Pelicans'
    elif x=='New Orleans/Oklahoma City Hornets':
        return 'New Orleans Pelicans'
    elif x=='Charlotte Bobcats':
        return 'Charlotte Hornets'
    elif x=='Vancouver Grizzlies':
        return 'Memphis Grizzlies'
    elif x=='NO/Oklahoma City Hornets':
        return 'New Orleans Pelicans'
    elif x=='LA Clippers':
        return 'Los Angeles Clippers'
    else:
        return x

# %%
teamDf.team = teamDf.team.apply(lambda x: changedTeam(x))
len(teamDf.team.unique())

# %%
plyDf = plyDf.drop(["obbs"],axis=1)
plyDf

# %%
teamDf.rename(columns={'obbs':'team_obbs'}, inplace=True)

# %%
groupDf = plyDf.groupby(by=["season", "team", "position"]).mean()
groupDf

# %%
pd.set_option('display.max_rows', None)
groupDf
# %%
plyDf.groupby(by=["season","team"]).count()
# %%
dd = pd.read_csv("data\player_stats.csv")
dd.groupby(by=["season","team"]).count()
# %%
for season in dd.season.unique():
    print(len(dd[dd.season==season]))
    
# %%
