#%%
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

# %%
# rmse 함수 정의
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 


# 스케일링(정규화) 함수
def mmsc(df):
    df1 = df.drop(["inflation_salary", "obbs", "pev", "player", "team", "position", "season"], axis=1)
    mmSc = MinMaxScaler()
    mmdf = pd.DataFrame(mmSc.fit_transform(df1), columns=df1.columns)
    cdf = pd.concat([df[["player", "team", "season", "position"]], mmdf], axis=1)
    resultDf = cdf[cdf.season==2023]

    return resultDf


# 트레이드 함수
def trade(df, outPlayer='', myTeam='', inPlayer='', oppTeam=''):
    df1 = df.copy()
    df1.loc[(df1.team==myTeam)&(df1.player==outPlayer), 'team']=oppTeam
    df1.loc[(df1.team==oppTeam)&(df1.player==inPlayer), 'team']=myTeam
    
    return df1


# 승률 예측 함수
def obbsPre(model, df, team=''):
    gDf = df.groupby(by=["season", "team", "position"]).mean()
    teamStatsDf = gDf.loc[(2023, team), :]
    features = teamStatsDf.to_numpy()
    features = features.reshape((1,5,34))
    obbsPred = model.predict(features).reshape((1,))[0]
    
    return obbsPred


# 기존 승률과 트레이드 이후 승률 출력 함수
def obbsChange(model, playerDf, teamDf, outPlayer='', myTeam='', inPlayer='', oppTeam=''):
    mdf = pd.merge(playerDf, teamDf, on=['season', 'team'], how='inner')
    gDf = mdf.groupby(by=['season', 'team', 'position']).mean()
    existObbs = gDf.loc[(2023, myTeam), :]['team_obbs'].to_numpy()[0] # 기존 승률
    
    mmdf = mmsc(playerDf)
    tradedDf = trade(mmdf, outPlayer, myTeam, inPlayer, oppTeam)
    obbsPred = obbsPre(model, tradedDf, myTeam)
    
    print(f"{myTeam}의 기존 승률 : {existObbs}")
    print(f"{myTeam}의 트레이드 이후 승률 : {obbsPred}")

    return existObbs, obbsPred


# # %%
# outPlayer="Isaiah Todd"
# myTeam="Washington Wizards"
# inPlayer="Udonis Haslem"
# oppTeam="Miami Heat"

# obbsChange(loaded_model, plyDf, teamDf, outPlayer, myTeam, inPlayer, oppTeam)

# %%
