#%%|
import pandas as pd
import numpy as np

from tensorflow.python.keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

# %%
df = pd.read_csv(r"./nonCorr.csv").drop(["Unnamed: 0"], axis=1)

# %%
# 포지션 별로 연봉의 차이가 있는지 확인
df.groupby('position').mean().inflation_salary

# %%
# 포지션 별로 연봉의 차이가 있으므로 포지션도 feature로 활용 --> one hot encoding 진행..
ohDf = pd.concat([df,pd.get_dummies(df.position)], axis=1)
ohDf = ohDf.drop(['position'], axis=1)
ohDf

# %%
# 이전 시즌 데이터에 다음 시즌의 연봉 매칭
seasons = ohDf.season.unique()

salaries = []
for season in seasons:
    if season!=2023:
        df1 = ohDf[ohDf.season==season]
        df2 = ohDf[ohDf.season==(season+1)][['player', 'inflation_salary']]
        mergeDf = pd.merge(df1, df2, on='player', how='inner').reset_index(drop=True)
        salaries.append(mergeDf)
    else:
        continue

salDf = pd.concat(salaries).reset_index(drop=True)
salDf = salDf.rename(columns={'inflation_salary_x' : 'present_salary', 'inflation_salary_y' : 'next_salary'})
salDf

# %%
df.isna().sum()
# %%
seasons = ohDf.season.unique()

for season in seasons:
    sameDf = ohDf[ohDf["season"]==season]
    print(f"<<<<< {season} >>>>>")
    print(sameDf.player.value_counts().sort_values(ascending=False), "\n")
# %%
test = ohDf[ohDf["season"]==2014]
test[test.player=="Tony Mitchell"]
# %%
test = ohDf.drop_duplicates()
test = test[test["season"]==2008]
test[test.player=="Marcus Williams"]
# %%
