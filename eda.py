#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#%%
def listRep(l=[], past='', now=''):
    try:
        idx = l.index(past)
        l[idx] = now
        return l
    except:
        return l

#%%
# 데이터 출처의 카테고리별로 feature들을 대분류 진행
playerPath = f"./data/cate/dd/players/"
teamPath = f"./data/cate/dd/teams/"

playerDict = {}
playerKeys = []
teamDict = {}
teamKeys = []
for (root, dir, files) in os.walk(playerPath):
    for file in files:
        df = pd.read_csv(playerPath+file).drop(['Unnamed: 0', '#'], axis=1)
        cols = df.columns.str.replace('\r\n',' ')\
            .str.replace(' ', '_')\
            .str.replace('__', '_')\
            .to_list()
        cols = listRep(cols, 'def_rtg', 'defrtg')
        cols = listRep(cols, 'opp_pts_off_to', 'opp_pts_off_tov')
        cols = listRep(cols, 'opp_2nd_pts', 'opp_pts_2nd_chance')
        cols = listRep(cols, 'opp_fbps', 'opp_pts_fb')
        cols.append('obbs') # 승률 컬럼 추가
        playerDict[file[7:-4]] = cols
        playerKeys.append(file[7:-4])
        
for (root, dir, files) in os.walk(teamPath):
    for file in files:
        df = pd.read_csv(teamPath+file).drop(['Unnamed: 0', '#'], axis=1)
        cols = df.columns.str.replace('\r\n',' ')\
            .str.replace(' ', '_')\
            .str.replace('__', '_')\
            .to_list()
        teamDict[file[5:-4]] = cols
        teamKeys.append(file[5:-4])

#%%
print("<<< player >>>")
for key in playerKeys:
    print(f"--------------{key}--------------")
    print(playerDict[key])

print("\n\n")

print("<<< team >>>")
for key in teamKeys:
    print(f"--------------{key}--------------")
    print(teamDict[key])

# %%
plyDf = pd.read_csv(f"./data/1_player_fin.csv")
plyDf.columns = plyDf.columns.str.replace(' ','_').to_list()
plyDf.columns.to_list()

#%%
teamDf = pd.read_csv(f"data/team_fin.csv")
teamDf.columns = teamDf.columns.str.replace(' ','_').to_list()
teamDf.columns.to_list()

#%%
# position에 8개의 결측치 확인
print(plyDf.isna().sum())

# 결측치가 존재하는 행 출력
plyDf[plyDf.position.isna()]

# posiotion이 결측치로 나온 선수 "Eddy Curry"는 C position으로 뛰었음.
plyDf = plyDf.fillna("C")
print(plyDf.isna().sum().sum())
plyDf[plyDf.player=="Eddy Curry"]["position"]

# position별로 몇 명이 존재하는지 확인
print(plyDf.position.value_counts())

# 승률 컬럼 추가
plyDf['obbs'] = plyDf[['gp', 'w']].apply(lambda x: x.w/x.gp, axis=1)

# %%
# category별 컬럼명들이 전체 데이터안에 모두 존재하는지 확인
for key in playerDict.keys():
    print(len(playerDict[key]))
    print(len(plyDf[playerDict[key]].columns))

for key in teamDict.keys():
    print(len(teamDict[key]))
    print(len(teamDf[teamDict[key]].columns))
    
# %%
plyDf[playerDict['advanced']]

# %%
plyDf[]

# %%
