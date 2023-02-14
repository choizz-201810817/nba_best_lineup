#%%
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

origin_df = pd.read_csv('./data/new_df2.csv')
origin_df = origin_df.drop(['index'],axis=1)
origin_df
#%%
position = list(origin_df['position'].unique())
position

# %%
def fix(df, team = None, position=None,season=None):
    fix_df = df[(df['position']==position)&(df['team']==team)&(df['season']==season)]
    fix_df = fix_df.reset_index(drop=True)
    return fix_df

#%%
fix(origin_df,'Boston Celtics','PF')



# # %%
# min(origin_df['pev'])
# %%
# 전체데이터 각 포지션별 pev 평균값 구하기
def pos_pev(df):
    
    position = ['PF', 'SG', 'PG', 'SF', 'C']
    pev = []
    for pos in position : 
        score = df[df['position']==pos]['pev'].mean()
        pev.append(np.round(score,2))
        
    return position, pev


# 현재시즌에서 특정팀의 각 포지션별 pev 평균값 구하기
def team_pev(df,team,season=2023):
    
    position = ['PF', 'SG', 'PG', 'SF', 'C']
    pev = []
    for pos in position : 
        score = df[(df['team']==team)&(df['position']==pos)&(df['season']==season)]['pev'].mean()
        pev.append(np.round(score,2))
        
    return position, pev


# 특정팀에서 '전체 포지션 pev 평균'보다 낮은 포지션을 찾고, 해당 포지션내에 최저 pev를 가진 선수 추출
def bad_pev(df, team, season=2023):
    position = ['PF', 'SG', 'PG', 'SF', 'C']
    all_value = pos_pev(df)[1]
    team_value = team_pev(df,team,season)[1]
    
    bad_pos=[]
    bad_pev=[]
    bad_player=[]
    for i in range(5):
        if team_value[i]<all_value[i]:
            pos = position[i]
            fix_df = df[(df['team']==team)&(df['position']==pos)&(df['season']==season)]
            pos_player = fix_df[fix_df['pev']==min(fix_df['pev'])]['player']
            player = []
            for j in pos_player:
                if j not in player:
                    player.append(j)
            
            bad_pos.append(pos)
            bad_pev.append(team_value[i])
            bad_player.append(player)
        
    return bad_pos,bad_pev,bad_player


# minmax 표준화 함수
def scale(df):
    scaler = MinMaxScaler()
    scale_need = df.drop(['player','team','inflation_salary', 'position','season', 'pev'], axis=1)
    no_scale_need = df[['player', 'position', 'inflation_salary', 'season', 'team', 'pev']]

    scaled =  pd.DataFrame(scaler.fit_transform(scale_need), columns=scale_need.columns )
    result_df = pd.concat([scaled,no_scale_need],axis=1)
    return result_df


# 팀과 플레이어를 넣으면 인덱스 반환해주는 함수
def indices(df,team,player):
    seq = df[(df['team']==team)&(df['player']==player)].index[0]
    return seq

def pos_stat(df,pos):
    if pos =='PF':
        result_df=df[['team','position','netrtg', 'poss', 'obbs', 'pf', 'def_ws']]
    elif pos =='PG':
        result_df=df[['team','position','netrtg', 'poss', 'def_ws', 'defrtg', '%pf']]
    elif pos =='SG':
        result_df=df[['team','position','netrtg','poss', 'obbs', 'pf', '%pf']]
    elif pos =='SF':
        result_df=df[['team','position','netrtg','poss', 'def_ws', 'defrtg', 'pf']]
    elif pos =='C':
        result_df=df[['team','position','netrtg', 'def_ws', 'poss', '%pf', 'pf']]
    return result_df


def reco_system(df, team, player, position, season=2023):
    scale_df = scale(df)
    seq = indices(df,team,player)
    pos_df = pos_stat(scale_df,position)
    gf = pos_df.groupby(['team','position']).mean()
    stat_df = pos_df.drop(['team','position'], axis=1)

    # 방출대상은 해당팀의 방출대상 포지션들에 대한 능력치 평균들 / 그외에는 각 선수들의 능력치 값들 => 코사인유사도로 비슷한 선수들 추리기
    set = []
    for i in range(len(df)):
        if i==seq:
            val = gf[gf.index==(team,position)].values[0]
        else:
            val = stat_df.iloc[i,:].values
        set.append(val)
        
    cosine_sim = cosine_similarity(set, set)
    sim_scores = list(enumerate(cosine_sim[seq])) # 방출대상의 해당팀,해당포지션 능력치 평균과의 유사도 가져옴
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # 유사도 높은순으로 정렬
    player_indices = [i[0] for i in sim_scores if i[1]>0.90] # 유사도가 특정 임계치 이상이면 가져옴
    player_indices = [i for i in player_indices if df['position'][i]==position] #해당포지션으로만 추리기
    player_indices = [i for i in player_indices if df['team'][i]!=team] #다른팀 선수들로만 추리기
    player_indices = [i for i in player_indices if df['season'][i]==season] #현재시즌 선수들로만 추리기
    # 해당 선수 연봉 +-20%이내로 추리기
    stand_salary = df[(df['team']==team)&(df['player']==player)&(df['season']==season)]['inflation_salary'].values
    upper = stand_salary*1.2
    downer = stand_salary*0.8
    choice_list = []
    for i in player_indices:
        player_salary = df['inflation_salary'][i]
        if (player_salary>=downer)&(player_salary<=upper):
            choice_list.append(i)
    trade_reco = df.iloc[choice_list].sort_values(by='pev',ascending=False)[:10] #pev 높은순으로 10명만 데려오기
    trade_reco.reset_index(drop=True,inplace=True)
    return trade_reco


# 포지션별 추천 선수
def player_reco(df,team,position,season=2023):
    bad_find = bad_pev(df, team, season)
    player_list = bad_find[2][bad_find[0].index(position)]
    player_num = len(player_list)
    
    reco_list = []
    for i in range(player_num):
        player = player_list[i]
        reco = reco_system(df, team, player, position, season=2023)
        reco_list.append(reco)
    return player_list, reco_list
    
    
# 선수에 대한 추천 시스템


# # 최종적으로 모든 취약포지션, 방출선수, 추천받을선수
# def all_reco(df,team,season=2023):
#     bad_find = bad_pev(df, team, season)
#     if bad_find is not None:
#         num = len(bad_find[0])
#         pos_reco_df=[]
#         for i in range(num):
#             pos = bad_find[0][i]
#             print('취약 포지션 : {0}\n방출 대상 : {1}'.format(pos,bad_find[2][i]))
#             recommend = reco(df,team,pos,season)
#             print(recommend)
            
#     #         pos_reco_df.append(recommend)
#     # return bad_find, pos_reco_df


#%%
bad_pev(origin_df, 'Toronto Raptors')

#%%
reco_system(origin_df,'Toronto Raptors','Juancho Hernangomez','PF',season=2023)

#%%
player_reco(origin_df,'Toronto Raptors','PF',season=2023)[1][0]










#%%
import pandas as pd
import numpy as np

# %%
scale_df = pd.read_csv('./data/filledPosition.csv')
scale_df.drop(['Unnamed: 0'],axis=1,inplace=True)
scale_df


# %%
drop_df = scale_df.drop(['player'],axis=1)
gdf = drop_df.groupby(['season','team','position']).mean()

gdf_trans = gdf.transpose()

season_num = scale_df['season'].unique()
team_num = scale_df['team'].unique()

#%%
# 시즌별, 팀별 포지션 5개 다 있는 개수 ::677
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
y




#%%
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

X_train, X_val, y_train, y_val = train_test_split(X_data, y_label, test_size=0.20)

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
print("\n Test Accuracy: %.4f" % (DNN_model.evaluate(X_val, y_val)[1]))

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


# %%
# CNN
# 데이터를 불러옴

X_reshape_data = X_data.reshape(X_data.shape[0],5,34,1)
X_train, X_val, y_train, y_val = train_test_split(X_reshape_data, y_label, test_size=0.20)

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
print("\n Test Accuracy: %.4f" % (CNN_model.evaluate(X_val, y_val)[1]))

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


# %%
a = player_reco(origin_df,'Toronto Raptors','PF',season=2023)[0][0]
# %%
b = origin_df[(origin_df['team']=='Toronto Raptors')&(origin_df['player']!=a)&(origin_df['season']==2023)]
# %%
pos_mean = b.groupby(['season','team','position']).mean()
# if len(pos_mean)==5:
pos_mean.drop(['inflation_salary','obbs','pev'],axis=1).values



# %%

bad_pev(origin_df, 'Boston Celtics', 2023)
#%%
scale_df = scale(origin_df)
trade_out_name = player_reco(scale_df,'Boston Celtics','PF',season=2023)[0][0]
trade_in_name = 'Dean Wade'
trade_in_team = 'Cleveland Cavaliers'
#%%
new_player = scale_df[(scale_df['team']=='Cleveland Cavaliers')&(scale_df['player']=='Dean Wade')&(scale_df['season']==2023)]
out_df = scale_df[(scale_df['team']=='Boston Celtics')&(scale_df['player']!=a)&(scale_df['season']==2023)]
out_after_df = pd.concat([out_df,new_player],axis=0).drop(['team','inflation_salary','obbs','pev'],axis=1)
out_after_df
# %%
group_df= out_after_df.groupby(['season','position']).mean()
X_test = group_df.values
X_test_re = X_test.reshape(1,5,34,1)
CNN_model.predict(X_test_re)
# %%
group_df
# %%
