#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFECV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import shap


# #%%
# ##### 포지션 예측 모델 개발 --> 원본 데이터에 존재하지 않는 position데이터를 채움
# df = pd.read_csv(r"./nonCorr.csv").drop(["Unnamed: 0"], axis=1)
# df

# # %%
# ## 타겟인 position을 labelencoding 진행
# lbEnc = LabelEncoder()
# df.position = lbEnc.fit_transform(df.position)
# df

# # %%
# X_set = df.drop(["player", "team", "season", "inflation_salary"], axis=1)
# y_set = df.position

# #%%
# ## X features MinMaxScaling 진행
# mmSc = MinMaxScaler()
# mmX = mmSc.fit_transform(X_set)

# X_set1 = pd.DataFrame(mmX, columns=X_set.columns)

# # %%
# X_train, X_test, y_train, y_test = train_test_split(X_set1, y_set, test_size=0.25)
# print(f"X_train's shape : {X_train.shape}\n")
# print(f"X_test's shape : {X_test.shape}\n")
# print(f"y_train's shape : {y_train.shape}\n")
# print(f"y_test's shape : {y_test.shape}")

# # %%
# ##### 최적의 예측 모델 찾기
# ## RandomForestClassifier
# model = RandomForestClassifier()

# params = {"max_depth" : [5, 10, 15, 20],
#           "n_estimators" : [100, 300, 500]}

# gscv = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=StratifiedKFold(n_splits=5), verbose=2)
# gscv.fit(X_train, y_train)


# print(gscv.best_estimator_)
# print(gscv.best_score_)
# em = gscv.best_estimator_
# pred = em.predict(X_test)
# accSc = accuracy_score(y_test, pred)
# print(f"{model.__class__.__name__}'s best_estimator acc score : {accSc}")


# # %%
# ## LGBMClassifier
# model1 = LGBMClassifier()

# params1 = {"max_depth" : [5, 10, 15, 20],
#           "n_estimators" : [100, 300, 500]}

# gscv1 = GridSearchCV(estimator=model1, param_grid=params1, scoring='accuracy', cv=StratifiedKFold(n_splits=5), verbose=2)
# gscv1.fit(X_train, y_train)


# print(gscv1.best_estimator_)
# print(gscv1.best_score_)
# em1 = gscv1.best_estimator_
# pred1 = em1.predict(X_test)
# accSc1 = accuracy_score(y_test, pred1)
# print(f"{model1.__class__.__name__}'s best_estimator acc score : {accSc1}")


# #%%
# ## XGBClassifier
# model2 = XGBClassifier()

# params2 = {"max_depth" : [5, 10, 15, 20],
#           "n_estimators" : [100, 300, 500]}

# gscv2 = GridSearchCV(estimator=model2, param_grid=params2, scoring='accuracy', cv=StratifiedKFold(n_splits=5), verbose=2)
# gscv2.fit(X_train, y_train)


# print(gscv2.best_estimator_)
# print(gscv2.best_score_)
# em2 = gscv2.best_estimator_
# pred2 = em2.predict(X_test)
# accSc2 = accuracy_score(y_test, pred2)
# print(f"{model2.__class__.__name__}'s best_estimator acc score : {accSc2}")


#%%
df = pd.read_csv("./data/obbs_data1.csv").drop(["index"], axis=1)
df.info()


#%%
# 데이터 스케일링 진행(정규화)
df1 = df.drop(["player", "team", "season", "position"], axis=1)

scaler = MinMaxScaler()
mmDf = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns)
mmDf1 = pd.concat([df[["player","team","season"]],mmDf,df[["position"]]], axis=1)
mmDf1

# 데이터 분리
train_set = mmDf1[mmDf1.position.notna()]
test_set = mmDf1[mmDf1.position.isna()]

# 타겟값 레이블링 진행
encoder = LabelEncoder()
train_set.position = encoder.fit_transform(train_set.position)
train_set.position

# train, validation 데이터 분리
Xset = train_set.drop(['position'], axis=1)
yset = train_set.position

X_train, X_val, y_train, y_val = train_test_split(Xset, yset, test_size=.2, random_state=64)
print(f"X_train shape : {X_train.shape}")
print(f"X_val shape : {X_val.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_val shape : {X_val.shape}")


# %%
X_train1 = X_train.drop(["player", "team", "season"], axis=1)
X_val1 = X_val.drop(["player", "team", "season"], axis=1)


#%%
model1 = RandomForestClassifier()
model1.fit(X_train1, y_train)
pred = model1.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")


# %%
model2 = LGBMClassifier()
model2.fit(X_train1, y_train)
pred = model2.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")


# %%
model3 = XGBClassifier()
model3.fit(X_train1, y_train)
pred = model3.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")


##### XGBClassifier의 정확도가 가장 높으므로 해당 모델 선택
#%%
##### Featrue importance 도출
explainer = shap.TreeExplainer(model3)
shap_values = explainer.shap_values(X_train1)

shap.summary_plot(shap_values, X_train1, max_display=40)


#%%
# SHAP을 통해 도출한 Feature importance가 높은 변수들만 가지고 다시 학습 진행..
mmDf2 = mmDf1[["player", "team", "season", "height", "oreb%", f"%ast", "dreb%", "%pts_fbps",
               "%3pm", f"%blk", "ast/to", f"2fgm_%uast", "%pf", "%pts_2pt_mr",
               "%tov", "stl%", "pitp", "age", "%pfd", "to_ratio", "position"]]


# 데이터 분리
train_set = mmDf2[mmDf2.position.notna()]
test_set = mmDf2[mmDf2.position.isna()]


# 레이블링
encoder = LabelEncoder()
train_set["position"] = encoder.fit_transform(train_set.position)


# 독립변수와 종속변수 분리 (feature와 target 분리)
Xset = train_set.drop(['position'], axis=1)
yset = train_set.position


# train, validation 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(Xset, yset, test_size=.2, random_state=32)
print(f"X_train shape : {X_train.shape}")
print(f"X_val shape : {X_val.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_val shape : {X_val.shape}")


# object 변수 제거
X_train1 = X_train.drop(["player", "team", "season"], axis=1)
X_val1 = X_val.drop(["player", "team", "season"], axis=1)


# 하이퍼 파라미터 튜닝
params = {"max_depth" : [5, 10, 30, 50],
          "n_estimators" : [100, 300, 500, 700, 1000]}

model = XGBClassifier()

gscv = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=3, verbose=2)
gscv.fit(X_train1, y_train)

# 튜닝 결과 확인
print(gscv.best_estimator_)
print(gscv.best_score_)


#%%
xgClf = XGBClassifier(n_estimators=100, max_depth=30)
xgClf.fit(X_train1, y_train)
pred = xgClf.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"정확도 : {acc}")

#%%
##### -> Feature selection 진행 (RFE 기법)

# Data set 생성
Xset1 = Xset.drop(["player", "team", "season"], axis=1)

# XGBClassifier 선택
model = XGBClassifier()

# REFCV로 Feature들을 반복적으로 제거해가면서 학습/평가 수행
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5),
              scoring='accuracy', verbose=2)
rfecv.fit(Xset1, yset)


#%%
# RFE 기법으로 선택된 변수의 수와 변수들
print("Optimal number of features : %d" % rfecv.n_features_)
print("Selected features : ", Xset1.columns[rfecv.support_])

#%%
selectedCols = Xset1.columns[rfecv.support_]

#%%
mmDf2 = mmDf1[selectedCols]
mmDf2 = pd.concat([mmDf1[["player","team","season"]], mmDf2, mmDf1["position"]], axis=1)
mmDf2

#%%
train_set = mmDf2[mmDf2.position.notna()]
test_set = mmDf2[mmDf2.position.isna()]

#%%
## position labeling 진행

encoder = LabelEncoder()
train_set.position = encoder.fit_transform(train_set.position)
train_set.position

#%%
Xset = train_set.drop(['position'], axis=1)
yset = train_set.position

X_train, X_val, y_train, y_val = train_test_split(Xset, yset, test_size=.2, random_state=32)
print(f"X_train shape : {X_train.shape}")
print(f"X_val shape : {X_val.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_val shape : {X_val.shape}")

#%%
X_train1 = X_train.drop(["player", "team", "season"], axis=1)
X_val1 = X_val.drop(["player", "team", "season"], axis=1)

#%%
# 하이퍼 파라미터 튜닝
Xset1 = Xset.drop(["player", "team", "season"], axis=1)
yset1 = yset

params = {"max_depth" : [5, 10, 30, 50],
          "n_estimators" : [100, 300, 500, 700, 1000]}

lgbm = LGBMClassifier()

gscv = GridSearchCV(estimator=lgbm, param_grid=params, scoring='accuracy', cv=3, verbose=2)
gscv.fit(Xset1, yset1)

# %%
# 튜닝 결과 확인
print(gscv.best_estimator_)
print(gscv.best_score_)

# %%
tunedModel = LGBMClassifier(max_depth=10, n_estimators=1000)
tunedModel.fit(X_train1, y_train)
pred = tunedModel.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")

# %%
model2 = LGBMClassifier(max_depth=20, n_estimators=500)
model2.fit(X_train1, y_train)
pred = model2.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")

# %%
model3 = XGBClassifier(max_depth=20, n_estimators=500)
model3.fit(X_train1, y_train)
pred = model3.predict(X_val1)
acc = accuracy_score(y_val, pred)
print(f"포지션 예측 모델 validation accuracy : {acc}")

# %%
##### validation accuracy가 가장 높은 xgboost를 사용
##### feature importance 추출
explainer = shap.TreeExplainer(model3)
shap_values = explainer.shap_values(X_train1)

shap.summary_plot(shap_values, X_train1, max_display=40)

# %%
