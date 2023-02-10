#%%
from bs4 import BeautifulSoup
from selenium import webdriver

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import pandas as pd
import numpy as np

#%%
# 테이블의 body부분 크롤링
def body(driver):
    stats = []

    while (True):
        driver.execute_script("window.scrollTo(0, 350)") # 버튼이 보이는 곳까지 스크롤 내리기
        driver.implicitly_wait(7) # 테이블 생성 기다리기

        nextButton = driver.find_element(By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[5]/button[2]')
        table = driver.find_element(By.XPATH, f'//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table')
        
        # 테이블 body 가져오기
        tbody = table.find_element(By.TAG_NAME, "tbody")

        # 테이블 행 모두 가져오기
        rows = tbody.find_elements(By.TAG_NAME, "tr")

        # 각 행에서 값 모두 뽑아서 저장
        for row in rows:
            values = row.find_elements(By.TAG_NAME, "td")
            stat = [i.text for i in values]

            stats.append(stat) # 각 행이 리스트로 묶여 stats 리스트에 append됨(2차원 list 생성)

        if nextButton.is_enabled():
            nextButton.click()
        else:
            break

    return stats

# %%
generals = ["traditional", "advanced", "misc", "scoring", "usage", "defense", "estimated-advanced"]

# seasons = ["1999-00", "2000-01", "2001-02", "2002-03", "2003-04", "2004-05", "2005-06"
# , "2006-07", "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14"
# , "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"
# , "2022-23"]

seasons = ["2022-23"]

playerDfs = []

driver = webdriver.Chrome()
for general in generals:
    url = f"https://www.nba.com/stats/players/{general}?Season="

    for season in seasons:
        driver.get(url+season)
        driver.refresh()
        driver.implicitly_wait(15)
        table = driver.find_element(By.XPATH, f'//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table')

        # 테이블 헤드 가져오기
        thead = table.find_element(By.TAG_NAME, "thead")
        head_row = thead.find_element(By.TAG_NAME, "tr")
        cols = head_row.find_elements(By.TAG_NAME, "th")

        columns = []
        for col in cols:
            columns.append(col.text)
        while ('' in columns):
            columns.remove('') # ''로 나오는 컬럼명 없애기
        if ' ' in columns:
            columns = columns[1:]
        columns.insert(0,'#')

        # 테이블 바디 가져오기
        stats = body(driver)

        df = pd.DataFrame(stats, columns=columns)
        df["season"] = "'"+season[2:]
        df.columns = df.columns.str.lower()
        playerDfs.append(df)



# %%
for i, general in enumerate(generals):
    df = pd.concat(playerDfs[i*24:(i+1)*24])
    df.to_csv(f'./data/{general}/player_{general}.csv')


#%%
# playerDfs1 = playerDfs[:24]
# # %%
# for df,season in zip(playerDfs1, seasons):
#     df.to_csv(f"./data/misc/player_stats/player_{season}.csv")
# # %%
# df = pd.concat(playerDfs1).reset_index(drop=True)
# df.to_csv('./data/player_misc.csv')

# %%
for i, df in enumerate(playerDfs):
    if i==0:
        mdf2 = df
    elif i==6:
        continue
    else:
        mdf2 = pd.merge(mdf2, df, on=["player", "team", "age", "season"])
        
        # 중복된 컬럼에서 _y로 병합된 컬럼은 제거
        dropCols = mdf2.filter(regex="_y").columns
        mdf2 = mdf2.drop(dropCols, axis=1)

        # _x로 병합된 컬럼명은 _x를 제외한 나머지 부분 가져오기
        mdf2.columns = [col[:-2] if "_x" in col else col for col in mdf2.columns]

#%%
mdf2 = pd.merge(mdf2, playerDfs[6], on=['player', 'season', 'gp'], how='inner')
dropCols = mdf2.filter(regex="_y").columns
mdf2 = mdf2.drop(dropCols, axis=1)
mdf2.columns = [col[:-2] if "_x" in col else col for col in mdf2.columns]
mdf2

# %%
mdf2.to_csv(f'./data/player_stats_2023.csv')

#%%
df2023 = pd.read_csv(f'./data/player_stats_2023.csv').drop(["Unnamed: 0"], axis=1)
df2023.columns.tolist()

# %%
oriDf = pd.read_csv("./data/player_stats.csv").drop(["Unnamed: 0"], axis=1)
oriDf.columns = oriDf.columns.str.replace('\r','')
oriCols = oriDf.columns.tolist()

#%%
# 컬럼 순서 맞추기
df2023 = df2023[oriCols]

# %%
oriDf1 = oriDf.drop(oriDf[oriDf.season=="'22-23"].index, axis=0)
resultDf = pd.concat([oriDf1, df2023]).reset_index(drop=True)
resultDf
# %%
resultDf.to_csv("./data/player_stats.csv")

#%%




















#%%
# bios 크롤링
# seasons = ["1999-00", "2000-01", "2001-02", "2002-03", "2003-04", "2004-05", "2005-06"
# , "2006-07", "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14"
# , "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"
# , "2022-23"]

seasons = ["2022-23"]

biosDfs = []

driver = webdriver.Chrome()

url = f"https://www.nba.com/stats/players/bio?Season="

for season in seasons:
    driver.get(url+season)
    driver.refresh()
    driver.implicitly_wait(15)
    table = driver.find_element(By.XPATH, f'//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table')

    # 테이블 헤드 가져오기
    thead = table.find_element(By.TAG_NAME, "thead")
    head_row = thead.find_element(By.TAG_NAME, "tr")
    cols = head_row.find_elements(By.TAG_NAME, "th")

    columns = []
    for col in cols:
        columns.append(col.text)
    while ('' in columns):
        columns.remove('') # ''로 나오는 컬럼명 없애기

    # 테이블 바디 가져오기
    stats = body(driver)

    df = pd.DataFrame(stats, columns=columns)
    df["season"] = "'"+season[2:]
    df.columns = df.columns.str.lower()
    biosDfs.append(df)

# %%
df = pd.concat(biosDfs).reset_index(drop=True)

# %%
def ft2cm(height=""):
    idx = height.index('-')
    ft = int(height[:idx])
    inch = int(height[idx+1:])
    cm = np.round(((ft*30.48)+(inch*2.54)), 2)
    return cm

#%%
df = df[df.height.apply(lambda x: False if x=='' else True)].reset_index(drop=True)
df.height = df.height.apply(lambda x: ft2cm(x))
df.weight = df.weight.astype("float64")

# %%
df.to_csv(f'./data/player_bios_2023.csv')

#%%
bios2023 = pd.read_csv(f'./data/player_bios_2023.csv').drop(["Unnamed: 0"], axis=1)

# %%
bios = pd.read_csv("./data/player_bios.csv").drop(["Unnamed: 0"], axis=1)
biosCols = bios.columns.tolist()

#%%
bios2023 = bios2023[biosCols]

#%%
bios1 = bios.drop(bios[bios.season=="'22-23"].index, axis=0)
biosDf = pd.concat([bios1, bios2023]).reset_index(drop=True)
biosDf

# %%
biosDf.to_csv("./data/player_bios.csv")
