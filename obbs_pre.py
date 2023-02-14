#%%
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


from tensorflow.python.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.python.keras import Model, Sequential, regularizers
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from keras.initializers.initializers_v1 import HeNormal
from tensorflow.python.keras.metrics import RootMeanSquaredError
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

# %%
teamDf = pd.read_csv("./data/teamObbs.csv").drop(["Unnamed: 0"], axis=1)
teamDf.season = teamDf.season.astype("str")

plyDf = pd.read_csv("./data/NotNull.csv").drop(["Unnamed: 0", "inflation_salary", "pev"], axis=1)

#%%
tempDf = plyDf.drop(["player", "team", "season", "position"], axis=1)

mmSc = MinMaxScaler()
mmDf = pd.DataFrame(mmSc.fit_transform(tempDf), columns=tempDf.columns)
plyDf = pd.concat([plyDf[["player", "team", "season", "position"]], mmDf], axis=1)
plyDf.season = plyDf.season.astype("str")

#%%
obbsDf = pd.merge(plyDf, teamDf, on=["team", "season"], how="inner")
groupDf = obbsDf.groupby(by=["season", "team", "position"]).mean()
groupDf

#%%
# 5개의 포지션이 모두 존재하는 팀 데이터만 가져오기
for season in plyDf.season.unique():
    for team in plyDf.team.unique():
        try:
            teamPosesNum = len(groupDf.loc[(season, team), :])
            if teamPosesNum<5:
                groupDf = groupDf.drop((season, team))
            else:
                continue
        except:
            continue
            
# %%
# 데이터셋 생성
pd.set_option('display.max_rows', 10)
teamDfs = []
for season in plyDf.season.unique():
    for team in plyDf.team.unique():
        try:
            df = groupDf.loc[(season, team), :]
            teamDfs.append(df)
        except:
            continue

dataSet = pd.concat(teamDfs).reset_index(drop=True)
dataSet

#%%
# feature와 target 분리
X_set = dataSet.drop(["team_obbs"], axis=1)
y_set = dataSet.team_obbs

#%%
# 데이터 슬라이싱 (2차원 학습 데이터와 (1,)의 타겟 데이터)
team_stats = []
team_obbses = []
for i in range(int(len(dataSet)/5)):
    team_stat_array = X_set.iloc[i*5:(i+1)*5, :].to_numpy()
    team_obbs = y_set.iloc[i*5:(i+1)*5].to_numpy()[0]
    team_stats.append(team_stat_array)
    team_obbses.append(team_obbs)

features = np.array(team_stats)
targets = np.array(team_obbses)

print(f"features's shape : {features.shape}")
print(f"targets's shape : {targets.shape}")

# %%
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=32)
print(f"X_train's shape : {X_train.shape}")
print(f"X_val's shape : {X_val.shape}")
print(f"y_train's shape : {y_train.shape}")
print(f"y_val's shape : {y_val.shape}")

# %%
# rmse 함수 정의
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
# %%
def cnnModel(X_train, X_val, y_train, y_val, HIDDEN_UNITS, KERNEL_SIZE, INITIALIZER, NORM, opti, EPOCHS, BATCH_SIZE, checkpoint):
    model = Sequential()
    model.add(Conv1D(HIDDEN_UNITS, kernel_size=KERNEL_SIZE, input_shape=X_train.shape[1:], activation='elu', kernel_initializer=INITIALIZER))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='elu', kernel_regularizer=NORM))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer=opti, loss=rmse, metrics=[RootMeanSquaredError()])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val), 
                        epochs=EPOCHS, 
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint])
    
    return model, history

# %%
HIDDEN_UNITS = 256
EPOCHS = 2500
BATCH_SIZE = 64
opti = Nadam(learning_rate=0.00005)
NORM = regularizers.l2(0.1)
INITIALIZER = HeNormal()
KERNEL_SIZE = 3

save_path = './model_save/'+'{epoch:03d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model, history = cnnModel(X_train, X_val, y_train, y_val, HIDDEN_UNITS, KERNEL_SIZE, INITIALIZER, NORM, opti, EPOCHS, BATCH_SIZE, checkpoint)

# %%
# 모델 평가
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# %%
# 모델 평가2
plt.figure(figsize=(12,8))
plt.plot(np.arange(300,2500), history.history['loss'][300:])
plt.plot(np.arange(300,2500), history.history['val_loss'][300:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# %%
# 모델 평가3
plt.figure(figsize=(12,8))
plt.plot(np.arange(800,2500), history.history['loss'][800:])
plt.plot(np.arange(800,2500), history.history['val_loss'][800:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#%%
# 모델 로드
hdf5_path = './model_save/2438-0.0660.hdf5'
loaded_model = load_model(hdf5_path, custom_objects={'rmse': rmse})

# %%
# 모델 평가 (validation data로 rmse 계산)
pred = loaded_model.predict(X_val)
pred1 = pred.reshape((138,))

ses = []
for p, y in zip(pred1, y_val):
    se = np.square(y-p)
    ses.append(se)

val_rmse = np.sqrt(np.mean(se))

print(f"model's rmse of validation : {val_rmse}")

# %%
# 트레이드 함수
def trade(df, outPlayer='', myTeam='', inPlayer='', oppTeam=''):
    df1 = df.copy()
    df1[(df1.team==myTeam)&(df1.player==outPlayer)].team=oppTeam
    df1[(df1.team==oppTeam)&(df1.player==inPlayer)].team=myTeam
    
    return df1

# 승률 예측 함수
def obbsPre(model, df, team=''):
    gDf = df.groupby(by=["season", "team", "position"]).mean()
    teamStatsDf = gDf.loc[("2023", team), :]
    features = teamStatsDf.to_numpy()
    features = features.reshape((1,5,34))
    obbsPred = model.predict(features).reshape((1,))[0]
    
    return obbsPred

# 기존 승률과 트레이드 이후 승률 출력 함수
def obbsChange(model, playerDf, teamDf, outPlayer='', myTeam='', inPlayer='', oppTeam=''):
    mdf = pd.merge(playerDf, teamDf, on=['season', 'team'], how='inner')
    gDf = mdf.groupby(by=['season', 'team', 'position']).mean()
    existObbs = gDf.loc[('2023', myTeam), :]['team_obbs'].to_numpy()[0] # 기존 승률
    
    tradedDf = trade(playerDf, outPlayer, myTeam, inPlayer, oppTeam)
    obbsPred = obbsPre(model, tradedDf, myTeam)
    
    print(f"{myTeam}의 기존 승률 : {existObbs}")
    print(f"{myTeam}의 트레이드 이후 승률 : {obbsPred}")

# %%
outPlayer="Isaiah Todd"
myTeam="Washington Wizards"
inPlayer="Udonis Haslem"
oppTeam="Miami Heat"

obbsChange(loaded_model, plyDf, teamDf, outPlayer, myTeam, inPlayer, oppTeam)

# %%
