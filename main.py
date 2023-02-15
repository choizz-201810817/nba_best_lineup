#%%
from reco_model_mark2 import playerRecommend
from obbs_pre_module import obbsChange, rmse, trade

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.metrics import RootMeanSquaredError
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

# %%
plyDf = pd.read_csv(f"./data/NotNull.csv").drop(["Unnamed: 0"], axis=1)
teamDf = pd.read_csv("./data/teamObbs.csv").drop(["Unnamed: 0"], axis=1)

hdf5_path = './model_save/temp/obbs/0214_1247/2438-0.0660.hdf5'
loaded_model = load_model(hdf5_path, custom_objects={'rmse': rmse})

# %%
# Utah Jazz
team=input('원하는 팀을 입력하십시오 : ')
emissionPlyer, recoList = playerRecommend(df=plyDf, team=team)

emPly = emissionPlyer.player.values[0]
print(f"방출 대상 선수 : {team}의 {emPly}\n\n")

for i, (recoTeam, recoPly) in enumerate(zip(recoList.team, recoList.player)):
    print(f"영입 추천 선수 {i+1} : {recoTeam}의 {recoPly}")

#%%
myTeam="Utah Jazz"
outPlayer="Rudy Gay"
oppTeam="New York Knicks"
inPlayer="Cam Reddish"

existObbs, obbsPred = obbsChange(loaded_model, plyDf, teamDf, outPlayer, myTeam, inPlayer, oppTeam)

# %%
