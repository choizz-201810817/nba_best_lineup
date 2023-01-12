#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

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
            .str.replace('\n',' ')\
            .str.replace(' ', '_')\
            .str.replace('__', '_')\
            .to_list()
        cols = listRep(cols, 'ast_ratio', 'est._ast_ratio')
        cols = listRep(cols, 'def_rtg', 'defrtg')
        cols = listRep(cols, 'opp_pts_off_tov', 'opp_pts_off_to')
        cols = listRep(cols, 'opp_pts_2nd_chance', 'opp_2nd_pts')
        cols = listRep(cols, 'opp_pts_fb', 'opp_fbps')
        cols = listRep(cols, 'opp_pts_paint', 'opp_pitp')
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
teamDf = pd.read_csv(f"data/team_fin.csv").drop(['Unnamed: 0'], axis=1)
teamDf.columns = teamDf.columns.str.replace(' ','_')\
            .str.replace('\n','_')\
            .str.replace('__','_').to_list()
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
print("<<< player >>>")
for key in playerDict.keys():
    print(len(playerDict[key]))
    print(len(plyDf[playerDict[key]].columns))
    
print("<<< team >>>")
for key in teamDict.keys():
    print(len(teamDict[key]))
    print(len(teamDf[teamDict[key]].columns))
    

#%%
test = playerDict['traditional']
test.append('pace')
test.append('pie')
test.append('poss')
test

# %%
testDf = plyDf[test]
testDf1 = testDf[testDf['gp']>=5].iloc[:,2:]
testDf1 = testDf1.drop(["season"],axis=1)
testDf1

# %%
testDf1.info()

#%%
mm_sc = MinMaxScaler()
mmDf = pd.DataFrame(mm_sc.fit_transform(testDf1), columns=testDf1.columns)
mmDf = mmDf.drop(['w', 'l', '+/-'],axis=1)

# %%
# feature들의 상관관계 보기
mask = np.zeros_like(mmDf.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,10))
sns.heatmap(mmDf.corr(), mask=mask, cmap='RdYlBu_r', linewidths=1)

# %%
mmDf.corr()['obbs'].sort_values(ascending=False)

#%%
X_set = mmDf.drop(['obbs'], axis=1)
y_set = mmDf.obbs

#%%
train_X, test_X, train_y, test_y = train_test_split(X_set, y_set, test_size=.25)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)

# %%
xgb = XGBRegressor()
xgb.fit(train_X, train_y)

# %%
featrue_imp = xgb.feature_importances_
plt.figure(figsize=(15,8))
plt.bar(train_X.columns, featrue_imp)
plt.show()

# %%
