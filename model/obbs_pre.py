#%%
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def scale(df):
    scaler = MinMaxScaler()
    scale_need = df.drop(['player','team','inflation_salary', 'position','season', 'pev'], axis=1)
    no_scale_need = df[['player', 'position', 'inflation_salary', 'season', 'team', 'pev']]

    scaled =  pd.DataFrame(scaler.fit_transform(scale_need), columns=scale_need.columns )
    result_df = pd.concat([scaled,no_scale_need],axis=1)
    return result_df

all_df = pd.read_csv('../data/new_df3.csv')
all_df.drop(['index'],axis=1,inplace=True)
scale_df = scale(all_df)
scale_df
# %%
drop_df = scale_df.drop(['player','inflation_salary','pev'],axis=1)
gdf = drop_df.groupby(['season','team','position']).mean()

gdf_trans = gdf.transpose()

season_num = scale_df['season'].unique()
team_num = scale_df['team'].unique()

#%%
# 시즌별, 팀별 포지션 5개 다 있는 개수 ::687
Z=[]
for i in season_num:
    for j in team_num:
        try:
            A = gdf_trans[i][j].transpose()
            if len(A)==5:
                Z.append(A)
            else:
                pass
        except:
            pass

len(Z)
#%% 
# X,y 정의

X=[]
y=[]
# yyy=[]
for i in season_num:
    for j in team_num:
        try:
            A = gdf_trans[i][j].transpose()
            if len(A)==5:
                data = A.drop(['obbs'],axis=1).values
                obbs = A.obbs.mean()
                # yyy.append(obbs)
                X.append(data)
                y.append(obbs)
                
            else:
                pass
        except:
            pass

#%%
X_data = np.array(X)
y_label = np.array(y)
#%%
# gdf_trans[2023]['Washington Wizards'].transpose().obbs.mean()
# len(X)
# gdf_trans[2000]['Atlanta Hawks'].transpose().drop(['obbs'],axis=1).values.shape
X_data.shape




#%%
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.20)

# DNN
DNN_model = Sequential()
DNN_model.add(Flatten(input_shape=(5,34)))
DNN_model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
DNN_model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
DNN_model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
DNN_model.add(Dense(1))

# 모델정의
DNN_model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
print(DNN_model.summary())


# 모델 학습
history = DNN_model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200)

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (DNN_model.evaluate(X_test, y_test)[1]))

# 검증셋과 학습셋의 오차 저장
y_vloss_DNN = history.history['val_loss']
y_loss_DNN = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss_DNN))
plt.figure(facecolor='white')
plt.plot(x_len, y_vloss_DNN, marker='.', c="red", label='DNN_Testset_loss')
plt.plot(x_len, y_loss_DNN, marker='.', c="blue", label='DNN_Trainset_loss')

# 그래프에 그리드를 주고 레이블 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 모델 저장하기
DNN_model.save('DNN_obbs_model.h5')





# %%
# CNN
# 데이터를 불러옴

X_reshape_data = X_data.reshape(X_data.shape[0],5,34,1)
X_train, X_test, y_train, y_test = train_test_split(X_reshape_data, y_label, test_size=0.20)

CNN_model = Sequential()
CNN_model.add(Conv2D(32, kernel_size=(3,3), input_shape=(5,34,1), activation='relu'))
CNN_model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Dropout(0.25)) 
CNN_model.add(Flatten())  # 컨볼루션 끝난것을 일반 노드로 연결하기 위해 FLATTEN
CNN_model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
CNN_model.add(Dense(1))

# 모델정의
CNN_model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'])
print(CNN_model.summary())

# 모델 학습
history = CNN_model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200)

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (CNN_model.evaluate(X_test, y_test)[1]))

# 검증셋과 학습셋의 오차 저장
y_vloss_CNN = history.history['val_loss']
y_loss_CNN = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss_CNN))
plt.figure(facecolor='white')
plt.plot(x_len, y_vloss_CNN, marker='.', c="red", label='CNN_Testset_loss')
plt.plot(x_len, y_loss_CNN, marker='.', c="blue", label='CNN_Trainset_loss')

# 그래프에 그리드를 주고 레이블 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 모델 저장하기
CNN_model.save('CNN_obbs_model.h5')



# %%
