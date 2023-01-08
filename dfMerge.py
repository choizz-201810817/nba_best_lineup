#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
# %%
plyPath = "./data/players/"
teamPath = "./data/teams/"

plyDfs = []
teamDfs = []
for (root, dir, files) in os.walk(plyPath):
    for file in files:
        df = pd.read_csv(plyPath+file).drop(["Unnamed: 0"], axis=1)
        plyDfs.append(df)

for (root, dir, files) in os.walk(teamPath):
    for file in files:
        df = pd.read_csv(teamPath+file).drop(["Unnamed: 0"], axis=1)
        teamDfs.append(df)

# %%
seasons = ["'99-00", "'00-01", "'01-02", "'02-03", "'03-04", "'04-05", "'05-06"
, "'06-07", "'07-08", "'08-09", "'09-10", "'10-11", "'11-12", "'12-13", "'13-14"
, "'14-15", "'15-16", "'16-17", "'17-18", "'18-19", "'19-20", "'20-21", "'21-22"
, "'22-23"]

plyerMergeDfs = []
for season in seasons:
    for i, df in enumerate(plyDfs):
        if i==0:
            mergeDf = df[df["season"]==season]
        else:
            mergeDf = pd.merge(mergeDf, df[df["season"]==season], on="player", how="inner")

            # 중복된 컬럼에서 _y로 병합된 컬럼은 제거
            dropCols = mergeDf.filter(regex="_y").columns
            mergeDf = mergeDf.drop(dropCols, axis=1)

            # _x로 병합된 컬럼명은 _x를 제외한 나머지 부분 가져오기
            mergeDf.columns = [col[:-2] if "_x" in col else col for col in mergeDf.columns]

    plyerMergeDfs.append(mergeDf)

#%%
plyFinalDf = pd.concat(plyerMergeDfs).reset_index(drop=True)
plyFinalDf.to_csv("./data/player_stats.csv")

# %%
# %%
seasons = ["'99-00", "'00-01", "'01-02", "'02-03", "'03-04", "'04-05", "'05-06"
, "'06-07", "'07-08", "'08-09", "'09-10", "'10-11", "'11-12", "'12-13", "'13-14"
, "'14-15", "'15-16", "'16-17", "'17-18", "'18-19", "'19-20", "'20-21", "'21-22"
, "'22-23"]

teamMergeDfs = []
for season in seasons:
    for i, df in enumerate(teamDfs):
        if i==0:
            mergeDf = df[df["season"]==season]
        else:
            mergeDf = pd.merge(mergeDf, df[df["season"]==season], on="team", how="inner")

            # 중복된 컬럼에서 _y로 병합된 컬럼은 제거
            dropCols = mergeDf.filter(regex="_y").columns
            mergeDf = mergeDf.drop(dropCols, axis=1)

            # _x로 병합된 컬럼명은 _x를 제외한 나머지 부분 가져오기
            mergeDf.columns = [col[:-2] if "_x" in col else col for col in mergeDf.columns]

    teamMergeDfs.append(mergeDf)
# %%
teamFinalDf = pd.concat(teamMergeDfs).reset_index(drop=True)
teamFinalDf.to_csv("./data/team_stats.csv")
# %%
