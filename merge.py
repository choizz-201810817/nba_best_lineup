#%%
import pandas as pd
import numpy as np

# %%
plydf = pd.read_csv("./data/player_stats.csv").drop(["Unnamed: 0"], axis=1)
biosdf = pd.read_csv("./data/player_bios.csv").drop(["Unnamed: 0"], axis=1)[["player", "team", "season", "age", "height", "weight"]]
nondf = pd.read_csv("./nonCorr.csv").drop(["Unnamed: 0"], axis=1)

# %%
plydf.columns = plydf.columns.str.replace("\n"," ").str.replace("  "," ").str.replace(" ","_")

# %%
plyBios = pd.merge(plydf, biosdf, on=["player", "team", "age", "season"], how='inner')

# %%
dropCols = [col for col in plyBios.columns if '_y' in col]

plyBios.columns = [col[:-2] if "_x" in col else col for col in plyBios.columns]
plyBios = plyBios.drop(dropCols, axis=1)
plyBios.columns.tolist()

# %%
plyBios.obbs = plyBios.w/plyBios.gp
plyBios.obbs

# %%
team1 = ['Atlanta Hawks', 'Cleveland Cavaliers', 'Minnesota Timberwolves',
 'Boston Celtics', 'Charlotte Hornets', 'Utah Jazz', 'Dallas Mavericks',
 'Philadelphia 76ers', 'Seattle SuperSonics', 'Detroit Pistons',
 'Washington Wizards', 'Golden State Warriors', 'Vancouver Grizzlies',
 'Orlando Magic', 'Milwaukee Bucks', 'Los Angeles Lakers', 'Phoenix Suns',
 'Portland Trail Blazers', 'Houston Rockets', 'Indiana Pacers',
 'Toronto Raptors', 'Sacramento Kings', 'Denver Nuggets', 'Chicago Bulls',
 'New Jersey Nets', 'Los Angeles Clippers', 'Miami Heat', 'San Antonio Spurs',
 'New York Knicks', 'Memphis Grizzlies', 'New Orleans Hornets',
 'Charlotte Bobcats', 'NO/Oklahoma City Hornets', 'Oklahoma City Thunder',
 'Brooklyn Nets', 'New Orleans Pelicans']

team2 = ['ATL', 'CLE', 'MIN', 'BOS', 'CHH', 'UTA', 'DAL', 'PHI', 'SEA', 'DET',
         'WAS', 'GSW', 'VAN', 'ORL', 'MIL', 'LAL', 'PHX', 'POR', 'HOU', 'IND',
         'TOR', 'SAC', 'DEN', 'CHI', 'NJN', 'LAC', 'MIA', 'SAS', 'NYK', 'MEM',
         'NOH', 'CHA', 'NOK', 'OKC', 'BKN', 'NOP']

# %%
teams = []
for team in plyBios.team:
    idx = team2.index(team)
    teams.append(team1[idx])

plyBios.team = teams
# %%
def changedTeam(x=''):
    if x=='New Jersey Nets':
        return 'Brooklyn Nets'
    elif x=='New Orleans Hornets':
        return 'New Orleans Pelicans'
    elif x=='Charlotte Bobcats':
        return 'Charlotte Hornets'
    elif x=='Vancouver Grizzlies':
        return 'Memphis Grizzlies'
    elif x=='NO/Oklahoma City Hornets':
        return 'New Orleans Pelicans'
    else:
        return x
    
plyBios.team = plyBios.team.apply(lambda x: changedTeam(x))
len(plyBios.team.unique())

#%%
plyBios.season = plyBios.season.apply(lambda x: "20"+x[-2:])
plyBios.season

# %%
nondf1 = nondf[["player", "team", "age", "season", "position", "inflation_salary"]]
nondf1.season = nondf1.season.astype(str)

nonCorrCols = nondf.columns

# %%
ddf = pd.merge(plyBios, nondf1, on=["player", "team", "age", "season"], how='left')
ddf["obbs"] = ddf.w/ddf.gp
ddf.obbs

# %%
resultDf = ddf[nonCorrCols]
resultDf

# %%
resultDf.to_csv("./data/allPlayer.csv")

# %%
