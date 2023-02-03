#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

import shap

# import eli5
# from eli5.sklearn import PermutationImportance

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
    
    return xCoor, yCoor

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


def perImp(df, pos, model):
    dataSet = df[df.position==pos]
    X = dataSet.drop(['position', '+/-'], axis=1)
    y = dataSet['+/-']
    model.fit(X,y)
    result = permutation_importance(model, X, y, n_repeats=30, random_state=0)
    sorted_result = result.importances_mean.argsort()
    features = X.columns[sorted_result]
    impoDf = pd.DataFrame(result.importances_mean[sorted_result], index=features, columns=['feature_importance']).\
        sort_values('feature_importance', ascending=False)

    return impoDf


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

<<<<<<< HEAD
# %%
plyDf = pd.read_csv(f"./data/1_player_fin.csv")
plyDf.columns = plyDf.columns.str.replace(' ','_').to_list()
plyDf.columns.to_list()
=======
#%%
plyDf = pd.read_csv(f"./data/ply_final.csv").drop(['Unnamed: 0'], axis=1)
plyDf.columns = plyDf.columns.str.replace('\r\n', '_')
plyDf.columns = plyDf.columns.str.replace(' _', '_')
plyDf.columns = plyDf.columns.str.replace(' ', '_')
plyDf.columns.tolist()
>>>>>>> f071735196e8f827404b02cc930e81987f681dab

#%%
teamDf = pd.read_csv(f"data/team_fin.csv").drop(['Unnamed: 0'], axis=1)
teamDf.columns = teamDf.columns.str.replace(' ','_')\
            .str.replace('\n','_')\
            .str.replace('\r','')\
            .str.replace('__','_').to_list()
teamDf.columns.to_list()

<<<<<<< HEAD
=======
#%% 
## 포지션 재 분류
plyDf.loc[plyDf.position=='G',"position"] = plyDf[plyDf.position=='G'].apply(lambda x: 'PG' if x.height<192 else 'SG', axis=1)
plyDf.loc[plyDf.position=='F',"position"] = plyDf[plyDf.position=='F'].apply(lambda x: 'SF' if x.height<204 else 'PF', axis=1)
plyDf.loc[plyDf.position=='GF',"position"] = 'SF'
plyDf
>>>>>>> f071735196e8f827404b02cc930e81987f681dab
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

#%%
teamDf.columns.to_list()

# %%
# category별 컬럼명들이 전체 데이터안에 모두 존재하는지 확인
print("<<< player >>>")
for key in playerDict.keys():
    if (len(playerDict[key])==len(plyDf[playerDict[key]].columns)):
        print(f"{key}의 컬럼 모두 존재")
    else:
        print(f"{key}의 컬럼 불일치")
    
print("<<< team >>>")
for key in teamDict.keys():
    if (len(teamDict[key])==len(teamDf[teamDict[key]].columns)):
        print(f"{key}의 컬럼 모두 존재")
    else:
        print(f"{key}의 컬럼 불일치")

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

# # %%
# impDf = pd.read_excel(r"./data/etc/feature_importance.xlsx")

# featImpUni = []
# for pos in impDf.columns:
#     features = pd.DataFrame(impDf[pos].unique())
#     featImpUni.append(features)

# featureImpDf = pd.concat(featImpUni, axis=1)
# featureImpDf.columns = impDf.columns
# featureImpDf.to_csv("./data/etc/feature_imp_unique.csv")


##### 포지션 target의 feature importance #####
# # feature importance about the position
# rfClf = RandomForestClassifier()
# for key in mmDfs.keys():
#     df = mmDfs[key]
#     model = RandomForestClassifier()
#     featureImp(df, key, model, target='position')

##### 카테고리별 전처리 끝 #####

# %%
##### margin을 target으로 한 feature importance 추출하기 #####
##### 카테고리에서 상관관계 분석을 통해 추출한 feature들로만 진행 #####
##### est는 추정치 스탯으로서 예측에 의미가 없고 복잡도만 높인다고 판단 -> 제거함 #####
nonCorrCols = ['player', 'defrtg', 'netrtg', 'ast/to', 'oreb%', 'dreb%', 'to_ratio', 'ts%', 'pace', 'pie', 'poss',\
    'stl%', f'%blk', 'def_ws',\
    'fbps', 'pitp', 'pf',\
    '%pts_2pt_mr', '%pts_fbps', '%pts_ft', '%pts_offto', f'2fgm_%uast', f'3fgm_%uast',\
    'ft%', '3p%', \
    '%3pm', f'%fta', f'%ast', '%tov', '%pf', '%pfd', '%pts',\
    'height', 'age', 'position', 'inflation_salary', 'season', 'obbs', '+/-']
# 'est._offrtg', 'est._defrtg', 'est._ast_ratio', 'est._oreb%', 'est._dreb%', 'est._to_ratio', 'est._usg%', 'est._pace',\
#  'blk', 'pfd', f'%reb', 'fg%', 'weight', 'dreb', 'offrtg', , f'%fga_2pt', '%pts_3pt'

nonCorrDf = plyDf1[nonCorrCols]
nonCorrDf['season'] = nonCorrDf['season'].apply(lambda x: '20'+x[-2:])
nonCorrDf

# %%
nonCorrDf.to_csv("./nonCorr.csv")

# %%
# nonCorrDf.season = nonCorrDf.season.astype('int')
nonCorrDf.info()

#%%
nonCorrDf1 = nonCorrDf.drop(['player', 'position', 'inflation_salary', 'season', '+/-'], axis=1)

mm_sc = MinMaxScaler()
mmNonCorrDf = pd.DataFrame(mm_sc.fit_transform(nonCorrDf1), columns=nonCorrDf1.columns)
mmNonCorrDf

#%%
mmNonCorrDf1 = pd.concat([mmNonCorrDf, nonCorrDf[['position', '+/-']]], axis=1)
mmNonCorrDf1

# %%
corrMap(mmNonCorrDf1, key='nonCorr')

# #%%
# mmTemp = mmNonCorrDf1.drop(['position'], axis=1)
# cols = mmTemp.columns

# corrList = []
# for col in cols:
#     temp = pd.DataFrame(mmTemp.corr()[col].sort_values(ascending=False))
#     corrList.append(temp)

# #%%
# # 각 feature별로 상관관계가 높은 순서대로 정렬한 dataframe을 csv로 저장
# for col, df in zip(cols, corrList):
#     col1 = col.replace('/','_')
#     df.to_csv(f'./{col1}.csv')

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
## feature importances by positions
positions = mmNonCorrDf1.position.unique()
columnList = []
importances = []

for pos in positions:
    model = LGBMRegressor()
    dataSet = mmNonCorrDf1[mmNonCorrDf1.position==pos]
    cols, imps = featureImp(dataSet=dataSet, key=pos, model=model, target='+/-')
    columnList.append(cols)
    importances.append(imps)

# %%
##### LGBMRegressor의 parameter 튜닝 #####
X = mmNonCorrDf1.drop(['position', '+/-'], axis=1)
y = mmNonCorrDf1['+/-']

model = LGBMRegressor()

params = {"learning_rate" : [0.001, 0.01, 0.1, 0.3, 0.5],
          "max_depth" : [25, 50, 75],
          "n_estimators" : [100, 300, 500]}

gscv = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=3, verbose=2)
gscv.fit(X, y)

# %%
print(gscv.best_estimator_)
print(gscv.best_score_)

### best estimator : max_depth=25 / n_estimators = 300 ###

#%%
##### shap 을 통해 feature importance 추출 #####
# lgbm의 dataset 생성

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=25)

# lgbm 학습
lgbm = LGBMRegressor(max_depth=25, n_estimators=300)
lgbm.fit(X_train, y_train)
pred = lgbm.predict(X_val)
print(np.round(np.sqrt(mean_squared_error(y_val, pred)), 4))

# %%
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_val)
# %%
shap.summary_plot(shap_values, X_val)
# %%
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[1,:], X_val.iloc[1,:])
# %%
shap.force_plot(explainer.expected_value, shap_values, X_val) 
# %%
shap.summary_plot(shap_values, X_val, plot_type = "bar")

<<<<<<< HEAD
# %%
df = pd.read_csv("./data/1_ply_final.csv")

# %%
df[df['team']=="Atlanta Hawks"]
# %%
=======


# %%
positions = nonCorrDf.position.unique()

for pos in positions:
    df = nonCorrDf[nonCorrDf['position'=='C']]
    df['new'] = -(2.37*df.age) + (2.05*df.netrtg) + (1.80*df.pitp) + (1.27*df.def_ws) + (1.17*df.poss) - (0.81*df['%pf']) - (0.57*df.defrtg) + (0.55*df['%pts']) - (0.47*df.pf) + (0.37*df['dreb%']) + (0.30*df['%pts_fbps']) + (0.06*df.obbs) + (0.04*df.pie) + (0.03*df['oreb%'])

>>>>>>> f071735196e8f827404b02cc930e81987f681dab
