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
seasons = ["1999-00", "2000-01", "2001-02", "2002-03", "2003-04", "2004-05", "2005-06"
, "2006-07", "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14"
, "2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"
, "2022-23"]

url = f"https://www.nba.com/stats/teams/traditional?Season="

driver = webdriver.Chrome()
teamDfs = []

for season in seasons:
    driver.get(url+season)
    driver.implicitly_wait(2)
    driver.execute_script("window.scrollTo(0, 400)")

    table = driver.find_element(By.XPATH, f'//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]/table')

    # 테이블 헤드 가져오기
    thead = table.find_element(By.TAG_NAME, "thead")
    head_row = thead.find_element(By.TAG_NAME, "tr")
    cols = head_row.find_elements(By.TAG_NAME, "th")

    columns = []
    for col in cols:
        columns.append(col.text)
    columns = columns[1:28]
    columns.insert(0,'#')

    driver.implicitly_wait(2)
    # 테이블 바디 가져오기
    stats = body(driver)

    df = pd.DataFrame(stats, columns=columns)
    teamDfs.append(df)

# %%
for df,season in zip(teamDfs, seasons):
    df.to_csv(f"./team_stats/{season}.csv")