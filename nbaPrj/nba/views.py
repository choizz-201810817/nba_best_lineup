from django.shortcuts import render

from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.
from reco_model_mark2 import playerRecommend
from obbs_pre_module import obbsChange, rmse, trade

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.metrics import RootMeanSquaredError
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')


plyDf = pd.read_csv(f"../data/NotNull_pev.csv").drop(["index"], axis=1)
teamDf = pd.read_csv("../data/teamObbs.csv").drop(["Unnamed: 0"], axis=1)

hdf5_path = '../model_save/temp/obbs/0214_1247/2438-0.0660.hdf5'
loaded_model = load_model(hdf5_path, custom_objects={'rmse': rmse})

teams = teamDf.team.unique().tolist()

def totalMeans(df, season=2023):
    positions = df.position.unique()

    posMeans = {}
    for pos in positions:
        posMean = df[(df['position']==pos)&(df['season']==season)].pev.mean()
        posMeans[pos] = posMean

    return posMeans


def teamMeans(df, team='', season=2023):
    tempDf = df[(df.team==team)&(df.season==season)]
    positions = tempDf.position.unique()

    posMeans = {}
    for pos in positions:
        posMean = tempDf[tempDf['position']==pos].pev.mean()
        posMeans[pos] = posMean
        
    return posMeans

## 포지션별 전체 선수 pev평균과 팀내 pev평균을 비교 후 전체 평균보다 낮은 포지션들을 낮은 순서대로 추출(오름차순 정렬)
def compareMeans(df, team='', season=2023):
    total = totalMeans(df, season)
    team = teamMeans(df, team, season)
    
    posDict = {}
    for pos in team.keys():
        if (total[pos] > team[pos]):
            posDict[pos] = team[pos]
    
    sortedPos = sorted(posDict.items(), key=lambda item: item[1])    
    positions = [pos[0] for pos in sortedPos]
    
    return positions

def playerRecommend(df, team='', pos='', season=2023, margin=0.2, threshold=0.99):    
    ## 방출 대상 포지션의 선수목록 추출
    emissionDf = df[(df['team']==team)&(df['position']==pos)&(df['season']==season)]

    ## 방출 대상 선수 선정 (해당 포지션에서 pev가 가장 낮은 선수)
    emissionDf = emissionDf.sort_values(by='pev', ascending=True).reset_index(drop=True)
    emissionPlayer = emissionDf.iloc[[0],:]
    
    ## 방출 대상 선수의 연봉 추출 및 +-20% 값 저장
    emiSal = emissionPlayer['inflation_salary']
    maxSal = emiSal.values[0] + (emiSal.values[0]*margin)
    minSal = emiSal.values[0] - (emiSal.values[0]*margin)
    
    ## 특정팀 외의 선수들 중 방출 대상 선수와 같은 포지션이면서 연봉 20% 이내의 선수들 불러오기
    targetPlysDf = df.drop(df[df.team==team].index, axis=0)
    targetPlysDf = targetPlysDf[(targetPlysDf['season']==season)&(targetPlysDf['position']==pos)
                                &(targetPlysDf['inflation_salary']<=maxSal)&(targetPlysDf['inflation_salary']>=minSal)]
    
    ## 방출 대상 포지션의 스탯들의 평균값 도출
    means = emissionDf.mean()
    meanDf = pd.DataFrame(means).T
    
    ## 트레이드 대상 포지션의 평균값과 targetPlyDf를 concat
    targetPlysDf1 = targetPlysDf[meanDf.columns]
    conDf = pd.concat([meanDf, targetPlysDf1], axis=0)
    
    ## 특정 포지션을 기준으로 팀내 평균과 다른 팀의 모든 선수들간의 능력치 유사도 계산 (팀 색깔 반영)
    mtxDf = conDf.drop(['season', 'inflation_salary', 'pev'], axis=1)
    cosineSim = cosine_similarity(mtxDf, mtxDf)
    cosDf = pd.DataFrame(cosineSim, index=mtxDf.index, columns=mtxDf.index)[0]
    cosDf = cosDf.drop([0], axis=0)
    
    ## pev 혹은 obbs가 높은 순서대로 선추 추천
    resultDf = pd.merge(targetPlysDf, cosDf, left_index=True, right_index=True, how='inner')
    resultDf = resultDf.rename(columns={0:'cosine_sim'})
    resultDf = resultDf.sort_values(by='pev', ascending=False)
    resultDf = resultDf[resultDf['cosine_sim']>=threshold].reset_index(drop=True)
    
    return emissionPlayer, resultDf

def index(request):
    return render(request, 'nba/index.html')

def ver1(request):
    if request.method == 'POST':
        team = request.POST.get('team', '')
        pos = request.POST.get('pos', '')
        poses = compareMeans(df=plyDf, team=team)
        
        if pos=='':
            return render(request, "nba/ver1_result1.html", {'teams' : [team], 'poses' : poses})
        else:
            emissionPlayer, resultDf = playerRecommend(plyDf, team, pos)
            emPlayer = emissionPlayer.player.values[0]
            recos = resultDf[["team", "player"]].values.tolist()
            
            return render(request, "nba/ver1_result2.html", {'teams' : [team],
                                                            'poses' : [pos],
                                                            'emPlayer' : emPlayer,
                                                            'recos' : recos}) 

    else:
        return render(request, 'nba/ver1.html', {'teams' : teams})

    
# emPly = emissionPlyer.player.values[0]
# print(f"방출 대상 선수 : {team}의 {emPly}\n\n")

# recoTeams = []
# recoPlys = []
# for recoTeam, recoPly in zip(recoList.team, recoList.player):
#     recoTeams.append(recoTeam)
#     recoPlys.append(recoPly)
    
# recoResult = {
#     'myTeam' : team,
#     'emissionPlyer' : emPly,
#     'recoTeams' : recoTeams,
#     'recoPlyers' : recoPlys
# }