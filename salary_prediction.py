#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier


from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Model, Sequential, regularizers
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from keras.initializers.initializers_v1 import HeNormal
from tensorflow.python.keras.metrics import RootMeanSquaredError
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import backend as K


import shap

#%%
df = pd.read_csv(r"./data/new_df3.csv").drop(["index"], axis=1)
df.drop(["pev"], axis=1, inplace=True)

# %%
df.isna().sum()

# %%
# 포지션이 연봉과 상관관계가 높다면 one-hot으로 바꾸어서 feature에 넣을 예정
encoder = LabelEncoder()
df["position_label"] = encoder.fit_transform(df.position)

# %%
def corrHeatMap(df):
    plt.figure(figsize=(30,30))

    mask = np.zeros_like(df.corr(), dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(df.corr(), cmap='RdYlBu_r', annot=True, mask=mask, linewidths=1, vmin = -1, vmax = 1)
    plt.show()

#%%
corrHeatMap(df)

#%%
# corrDf = df.corr()

# cols = []
# for col in corrDf.columns:
#     colNum = len(corrDf[(corrDf[col]>0.4)|(corrDf[col]<-0.4)])
#     cols.append((col, colNum))

# cols.sort(key=lambda x: -x[1])

# for col, num in cols:
#     df = df.drop([col], axis=1)
#     corrHeatMap(df)

# %%
highCorrCols = df.corr()["inflation_salary"].abs().sort_values().index

for col in highCorrCols:
    df = df.drop([col], axis=1)
    corrHeatMap(df)

# %%
# columns = ["netrtg", "ts%", "pie", "poss", "def_ws", "fbps", "pitp", "pf", f"3fgm_%uast", "ft%", 
#            f"%fta", f"%ast", "%tov", "%pf", "%pfd", "%pts", "age", "inflation_salary", "+/-"]
columns = ["netrtg", "pitp", f"3fgm_%uast", "ft%", "ast/to", "dreb%", "pie", "3p%",
           "%tov", "%pf", "%pfd", "age", "inflation_salary",]


df1 = df[columns]
corrHeatMap(df1)

#%%
## 정규화 진행
mdf = df1.drop(["inflation_salary"], axis=1)

scaler = MinMaxScaler()
mdf1 = pd.DataFrame(scaler.fit_transform(mdf), columns=mdf.columns)

mdf2 = pd.concat([mdf1, df1.inflation_salary], axis=1)
mdf2

# %%
conDf = pd.concat([df[["player", "team", "position"]], mdf2], axis=1)
conDf

# %%
train_set = conDf[conDf["inflation_salary"].notna()]
test_set = conDf[conDf["inflation_salary"].isna()]

#%%
X = train_set.drop(["player", "team", "position", "inflation_salary"], axis=1)
y = train_set["inflation_salary"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32)
print(f"X_train's shape : {X_train.shape}")
print(f"X_val's shape : {X_val.shape}")
print(f"y_train's shape : {y_train.shape}")
print(f"y_val's shape : {y_val.shape}")

# %%
# 연봉 예측 모델 설계
def salPredictModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, EPOCHS, opti, lossFunc, NORM, BATCH_SIZE, INITIALIZER, checkpoint):
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS, input_dim=INPUT_DIM, activation='elu', kernel_initializer=INITIALIZER))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='elu', kernel_regularizer=NORM))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    model.summary()
    model.compile(optimizer=opti, loss=lossFunc, metrics=[RootMeanSquaredError()])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val), 
                        epochs=EPOCHS, 
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint])
    
    return model, history

#%%
# rmse 함수 정의
def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# %%
HIDDEN_UNITS = 128
EPOCHS = 4000
BATCH_SIZE = 64
opti = Nadam(learning_rate=0.003)
lossFunc = rmse
NORM = regularizers.l2(0.1)
INPUT_DIM = 12
INITIALIZER = HeNormal()

save_path = './model_save/'+'{epoch:03d}-{val_loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model, history = salPredictModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, 
                                 EPOCHS, opti, lossFunc, NORM, BATCH_SIZE, INITIALIZER, checkpoint)



#%%
# 모델 평가
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
train_set["inflation_salary"].describe()
# %%
