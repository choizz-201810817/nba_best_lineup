#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

#%%
def listRep(l=[], past='', now=''):
    try:
        idx = l.index(past)
        l[idx] = now
        return l
    except:
        return l
    
    
def mmSc(df):
    df1 = df.drop(['player', 'team', 'season', 'position', 'obbs'], axis=1) # object drop / target drop
    
    mm_sc = MinMaxScaler()
    mmDf = pd.DataFrame(mm_sc.fit_transform(df1), columns=df1.columns)
    mmDf = mmDf.drop(['gp', 'w', 'l'], axis=1) # 타겟인 obbs(승률)을 생성할 때 사용한 컬럼들 제외
    mmDf = pd.concat([mmDf, df.obbs], axis=1) # target 다시 붙이기
    
    return mmDf
    
    
def featureImp(dataSet, key, model, target=''):
    if (target=='obbs') or (target=='+/-'):
        X = dataSet.drop([target, 'position'], axis=1)
        y = dataSet[target]
        model.fit(X, y)   
        featrue_imp = model.feature_importances_
        xCoor = X.columns[np.argsort(featrue_imp)[::-1]]
        yCoor = np.sort(featrue_imp)[::-1]
    else:
        X = dataSet.drop([target, 'obbs'], axis=1)
        y = dataSet[target]
        model.fit(X, y)   
        featrue_imp = model.feature_importances_
        xCoor = X.columns[np.argsort(featrue_imp)[::-1]]
        yCoor = np.sort(featrue_imp)[::-1]

    plt.figure(figsize=(30,18))
    # plt.bar(x=xCoor, y=yCoor)
    plt.title(key)
    sns.barplot(x=xCoor, y=yCoor)
    plt.show()
    

def corrMap(df, key=''):
    df = df.drop(['obbs'], axis=1)
    mask = np.zeros_like(df.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(20,17))
    plt.title(key)
    sns.heatmap(df.corr(), mask=mask, cmap='RdYlBu_r', linewidths=1, annot=True)
    plt.show()


def checkVif(df):
    # testDf = df.drop(['obbs', 'position'], axis=1)
    # VIF = pd.DataFrame()
    # VIF['features'] = testDf.columns
    # VIF['VIF'] = [vif(testDf, i) for i in range(testDf.shape[1])]
    VIF = pd.DataFrame()
    VIF['features'] = df.columns
    VIF['VIF'] = [vif(df, i) for i in range(df.shape[1])]
    return VIF


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
        if (file[7:-4]=='estimated-advanced'):
            cols.append('team')
        cols.append('position')
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
##### 카테고리별 eda 진행 #####
# 카테고리별 데이터 생성
# plyDf1 = plyDf[plyDf['gp']>=5].reset_index(drop=True) # 5경기 이상 뛴 선수들만 가져옴
plyDf1 = plyDf.reset_index(drop=True)

cateDfs = {}
for key in playerDict.keys():
    df = plyDf1[playerDict[key]]
    cateDfs[key] = df


#%%
# 카테고리별 데이터들을 모두 정규화 진행 (feature마다 단위가 다르기 때문)
mmDfs = {}
for key in cateDfs.keys():
    df = mmSc(cateDfs[key])
    df = pd.concat([df, plyDf1.position], axis=1)
    mmDfs[key] = df

#%%
for key in mmDfs.keys():
    corrMap(mmDfs[key], key)

#%%
vifs = []
for key in mmDfs.keys():
    vifDf = checkVif(mmDfs[key])
    vifs.append(vifDf)

#%%
vifs

#%%
# 가장 예측을 잘 하는 알고리즘 선택
rfRg = RandomForestRegressor(warm_start=False)
lnRg = LinearRegression()
xgb = XGBRegressor()
lgbm = LGBMRegressor()

models = [rfRg, lnRg, xgb, lgbm]

for model in models:
    modelName = model.__class__.__name__
    mses = []
    for key in mmDfs.keys():
        X_set = mmDfs[key].drop(['obbs', 'position'], axis=1)
        y_set = mmDfs[key].obbs

        X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=.25)
            
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mses.append(mse)
    
    # 7개의 카테고리에서 각 알고리즘들의 mse 평균
    mseMean = np.round(np.mean(mses), 4)
    print(f"<<< {modelName} >>>")
    print("mean of mse :", mseMean)

## 학습 결과
# <<< RandomForestRegressor >>>
# mean of mse : 0.0193
# <<< LinearRegression >>>
# mean of mse : 0.0211
# <<< XGBRegressor >>>
# mean of mse : 0.0208
# <<< LGBMRegressor >>>
# mean of mse : 0.0184

# LGBM이 가장 예측을 잘 한 것으로 나타남
# -> LGBM으로 feature importance를 뽑음

#%%
# 5개의 포지션이 7개의 카테고리에서 어떤 feature가 중요한지 확인
positions = plyDf1.position.unique()

posDfs = {}
for key in mmDfs.keys():
    for pos in positions:
        posDfs[key+'_'+pos] = mmDfs[key][mmDfs[key].position==pos]

print(posDfs.keys())

# %%
# feature importance about the obbs
for key in posDfs.keys():
    df = posDfs[key]
    model = LGBMRegressor()
    featureImp(df, key, model, target='obbs')


##### 카테고리별 전처리 끝 #####
# %%
impDf = pd.read_excel(r"./data/etc/feature_importance.xlsx")

featImpUni = []
for pos in impDf.columns:
    features = pd.DataFrame(impDf[pos].unique())
    featImpUni.append(features)

featureImpDf = pd.concat(featImpUni, axis=1)
featureImpDf.columns = impDf.columns
featureImpDf.to_csv("./data/etc/feature_imp_unique.csv")

# %%
##### 포지션 target의 feature importance #####
# # feature importance about the position
# rfClf = RandomForestClassifier()
# for key in mmDfs.keys():
#     df = mmDfs[key]
#     model = RandomForestClassifier()
#     featureImp(df, key, model, target='position')


# %%
##### margin을 target으로 한 feature importance 추출하기 #####
nonCorrCols = ['player', 'offrtg', 'defrtg', 'netrtg', 'ast/to', 'oreb%', 'dreb%', 'to_ratio', 'ts%', 'pace', 'pie', 'poss',\
    'dreb', 'stl%', f'%blk', 'def_ws',\
    'fbps', 'pitp', 'blk', 'pf', 'pfd',\
    f'%fga_2pt', '%pts_2pt_mr', '%pts_3pt', '%pts_fbps', '%pts_ft', '%pts_offto', f'2fgm_%uast', f'3fgm_%uast',\
    'fg%', '3p%', 'ft%',\
    '%3pm', f'%fta', f'%reb', f'%ast', '%tov', '%pf', '%pfd', '%pts',\
    'weight', 'height', 'age', 'position', 'inflation_salary', 'season', 'obbs', '+/-']
# 'est._offrtg', 'est._defrtg', 'est._ast_ratio', 'est._oreb%', 'est._dreb%', 'est._to_ratio', 'est._usg%', 'est._pace',\

nonCorrDf = plyDf1[nonCorrCols]
nonCorrDf['season'] = nonCorrDf['season'].apply(lambda x: '20'+x[-2:])
nonCorrDf
# %%
nonCorrDf.to_csv("./nonCorr.csv")

# %%
nonCorrDf.season = nonCorrDf.season.astype('int')
nonCorrDf.info()

#%%
nonCorrDf1 = nonCorrDf.drop(['player', 'position', 'inflation_salary', 'season', '+/-'], axis=1)

mm_sc = MinMaxScaler()
mmNonCorrDf = pd.DataFrame(mm_sc.fit_transform(nonCorrDf1), columns=nonCorrDf1.columns)
mmNonCorrDf
# %%
checkVif(mmNonCorrDf).to_csv('./vif.csv')

#%%
mmNonCorrDf1 = pd.concat([mmNonCorrDf, nonCorrDf[['position', '+/-']]], axis=1)
mmNonCorrDf1

# %%
corrMap(mmNonCorrDf1, key='nonCorr')

#%%
mmTemp = mmNonCorrDf1.drop(['position'], axis=1)
cols = mmTemp.columns

corrList = []
for col in cols:
    temp = pd.DataFrame(mmTemp.corr()[col].sort_values(ascending=False))
    corrList.append(temp)
    
#%%
for col, df in zip(cols, corrList):
    col1 = col.replace('/','_')
    df.to_csv(f'./{col1}.csv')

#%%
corrList[3]

# %%
rfRg = RandomForestRegressor(warm_start=False)
lnRg = LinearRegression()
xgb = XGBRegressor()
lgbm = LGBMRegressor()

models = [rfRg, lnRg, xgb, lgbm]

for model in models:
    modelName = model.__class__.__name__
    X_set = mmNonCorrDf1.drop(['+/-', 'position'], axis=1)
    y_set = mmNonCorrDf1['+/-']
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.25, random_state=25)
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.round(np.sqrt(mean_squared_error(y_test, pred)), 4)
    print(f"<<<<< {modelName} >>>>>")
    print("rmse :", rmse)
    
## LightGBM의 예측 결과가 가장 좋게 나옴.
# <<<<< RandomForestRegressor >>>>>
# rmse : 0.516
# <<<<< LinearRegression >>>>>
# rmse : 1.2901
# <<<<< XGBRegressor >>>>>
# rmse : 0.4815
# <<<<< LGBMRegressor >>>>>
# rmse : 0.4394

# %%
positions = mmNonCorrDf1.position.unique()

for pos in positions:
    model = LGBMRegressor()
    dataSet = mmNonCorrDf1[mmNonCorrDf1.position==pos]
    featureImp(dataSet=dataSet, key=pos, model=model, target='+/-')

# %%
