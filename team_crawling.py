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

    driver.execute_script("window.scrollTo(0, 200)")
    driver.implicitly_wait(7) # 테이블 생성 기다리기

    table = driver.find_element(By.XPATH, f'//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table')
    
    # 테이블 body 가져오기
    tbody = table.find_element(By.TAG_NAME, "tbody")

    # 테이블 행 모두 가져오기
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    # 각 행에서 값 모두 뽑아서 저장
    for row in rows:
        values = row.find_elements(By.TAG_NAME, "td")
        stat = [i.text for i in values]

        stats.append(stat)

    return stats

# %%
generals = ["four-factors", "misc", "defense", "estimated-advanced", "traditional", "advanced", "scoring"]
# generals = ["traditional", "advanced", "scoring"]

# seasons = ["1999-00", "2000-01", "2001-02", "2002-03", "2003-04", "2004-05", "2005-06"
# , "2006-07", "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14"
# , "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"
# , "2022-23"]

seasons = ["2022-23"]

teamDfs = []

driver = webdriver.Chrome()
for general in generals:
    url = f"https://www.nba.com/stats/teams/{general}?Season="

    for season in seasons:
        driver.get(url+season)
        driver.refresh() # 표 안뜨는 경우를 대비하여 새로고침
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
        teamDfs.append(df)

# %%
# for i, general in enumerate(generals):
#     df = pd.concat(teamDfs[i*24:(i+1)*24])
#     df.to_csv(f'./data/{general}/team_{general}.csv')

# teamDfs1 = teamDfs[:24]
# # %%
# for df,season in zip(teamDfs1, seasons):
#     df.to_csv(f"./data/misc/team_stats/team_{season}.csv")
# # %%
# df = pd.concat(teamDfs1).reset_index(drop=True)
# df.to_csv('./data/team_misc.csv')

# %%
teamMergeDfs = []

for i, df in enumerate(teamDfs):
    if i==0:
        mdf = df
    else:
        mdf = pd.merge(mdf, df, on=["team"])
        
        # 중복된 컬럼에서 _y로 병합된 컬럼은 제거
        dropCols = mdf.filter(regex="_y").columns
        mdf = mdf.drop(dropCols, axis=1)

        # _x로 병합된 컬럼명은 _x를 제외한 나머지 부분 가져오기
        mdf.columns = [col[:-2] if "_x" in col else col for col in mdf.columns]

mdf

# %%
team_df = pd.read_csv(r"./data/team_stats.csv").drop(["Unnamed: 0"], axis=1)
teamCols = team_df.columns.tolist()
teamCols

# %%
mdf = mdf[teamCols]
mdf

# %%
team_df1 = team_df.drop(team_df[team_df.season=="'22-23"].index)

# %%
teamResult = pd.concat([team_df1, mdf]).reset_index(drop=True)
teamResult

# %%
teamResult.to_csv(r"./data/team_stats.csv")

# %%
