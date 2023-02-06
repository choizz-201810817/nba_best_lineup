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
def fix(df, team=None, position=None):
    if team is None and position is None:
        fix_df = df
    elif team is None and position is not None:
        fix_df = df[df['position']==position]
    elif team is not None and position is None:
        fix_df = df[df['team']==team]
    else:
        fix_df = df[(df['position']==position)&(df['team']==team)]
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


# def reco(df,team,player,season=2023):
#     scale_df = scale(df)
#     seq = indices(df,team,player)
#     pos = df['position'][seq]

#     drop_df = scale_df.drop(['inflation_salary', 'season', 'pev'], axis=1)
#     gf = drop_df.groupby(['team','position']).mean()
#     stat_df = scale_df.drop(['player','team','position','inflation_salary','season','pev'], axis=1)

#     # 방출대상은 해당팀의 방출대상 포지션들에 대한 능력치 평균들 / 그외에는 각 선수들의 능력치 값들 => 코사인유사도로 비슷한 선수들 추리기
#     set = []
#     for i in range(len(df)):
#         if i==seq:
#             val = gf[gf.index==(team,pos)].values[0]
#         else:
#             val = stat_df.iloc[i,:].values
#         set.append(val)
        
#     cosine_sim = cosine_similarity(set, set)
#     sim_scores = list(enumerate(cosine_sim[seq])) # 방출대상의 해당팀,해당포지션 능력치 평균과의 유사도 가져옴
#     # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # 유사도 높은순으로 정렬
#     player_indices = [i[0] for i in sim_scores if i[1]>0.95] # 유사도가 특정 임계치 이상이면 가져옴
#     player_indices = [i for i in player_indices if df['position'][i]==pos] #해당포지션으로만 추리기
#     player_indices = [i for i in player_indices if df['position'][i]!=team] #다른팀 선수들로만 추리기
#     player_indices = [i for i in player_indices if df['season'][i]!=season] #현재시즌 선수들로만 추리기
#     trade_reco = origin_df.iloc[player_indices].sort_values(by='pev',ascending=False)[:10] #pev 높은순으로 10명만 데려오기
#     trade_reco.reset_index(drop=True,inplace=True)
#     return trade_reco


# 포지션별 추천 선수
def reco(df,team,position,season=2023):
    scale_df = scale(df)
    bad_find = bad_pev(df, team, season)
    player = bad_find[2][bad_find[0].index(position)][0]
    seq = indices(df,team,player)

    drop_df = scale_df.drop(['inflation_salary', 'season', 'pev'], axis=1)
    gf = drop_df.groupby(['team','position']).mean()
    stat_df = scale_df.drop(['player','team','position','inflation_salary','season','pev'], axis=1)

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
    # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # 유사도 높은순으로 정렬
    player_indices = [i[0] for i in sim_scores if i[1]>0.95] # 유사도가 특정 임계치 이상이면 가져옴
    player_indices = [i for i in player_indices if df['position'][i]==position] #해당포지션으로만 추리기
    player_indices = [i for i in player_indices if df['position'][i]!=team] #다른팀 선수들로만 추리기
    player_indices = [i for i in player_indices if df['season'][i]==season] #현재시즌 선수들로만 추리기
    trade_reco = origin_df.iloc[player_indices].sort_values(by='pev',ascending=False)[:10] #pev 높은순으로 10명만 데려오기
    trade_reco.reset_index(drop=True,inplace=True)
    return trade_reco


# 최종적으로 모든 취약포지션, 방출선수, 추천받을선수
def all_reco(df,team,season=2023):
    bad_find = bad_pev(df, team, season)
    if bad_find is not None:
        num = len(bad_find[0])
        pos_reco_df=[]
        for i in range(num):
            pos = bad_find[0][i]
            print('취약 포지션 : {0}\n방출 대상 : {1}'.format(pos,bad_find[2][i]))
            recommend = reco(df,team,pos,season)
            print(recommend)
            
    #         pos_reco_df.append(recommend)
    # return bad_find, pos_reco_df
    


#%%
bad_pev(origin_df, 'Toronto Raptors')

#%%
reco(origin_df,'Toronto Raptors','PF',season=2023)

# %%
all_reco(origin_df,'Toronto Raptors')























#%%
scale_df = scale(origin_df)
bad_find = bad_pev(origin_df, 'Toronto Raptors')
player = bad_find[2][bad_find[0].index('SF')][0]
seq = indices(origin_df, 'Toronto Raptors',player)

drop_df = scale_df.drop(['inflation_salary', 'season', 'pev'], axis=1)
gf = drop_df.groupby(['team','position']).mean()
stat_df = scale_df.drop(['player','team','position','inflation_salary','season','pev'], axis=1)

# 방출대상은 해당팀의 방출대상 포지션들에 대한 능력치 평균들 / 그외에는 각 선수들의 능력치 값들 => 코사인유사도로 비슷한 선수들 추리기
set = []
for i in range(len(origin_df)):
    if i==seq:
        val = gf[gf.index==('Toronto Raptors','SF')].values[0]
    else:
        val = stat_df.iloc[i,:].values
    set.append(val)
    
cosine_sim = cosine_similarity(set, set)
sim_scores = list(enumerate(cosine_sim[seq]))
player_indices = [i[0] for i in sim_scores if i[1]>0.95] # 유사도가 특정 임계치 이상이면 가져옴
# player_indices = [i for i in player_indices if origin_df['position'][i]=='SF'] #해당포지션으로만 추리기
# player_indices = [i for i in player_indices if origin_df['position'][i]!='Toronto Raptors'] #다른팀 선수들로만 추리기
# player_indices = [i for i in player_indices if origin_df['season'][i]==2023] #현재시즌 선수들로만 추리기
len(player_indices)

