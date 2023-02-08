#%%
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier


from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Model, Sequential, regularizers
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from keras.losses import SparseCategoricalCrossentropy
from keras.initializers.initializers_v1 import HeNormal


import shap

#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
import tensorflow as tf
tf.test.is_gpu_available()

# %%
df = pd.read_csv("./data/obbs_data1.csv").drop(["index"], axis=1)

df1 = df.drop(["player","team","season","position"], axis=1)

mmSc = MinMaxScaler()
df1 = pd.DataFrame(mmSc.fit_transform(df1), columns=df1.columns)
mmDf = pd.concat([df[["player","team","season"]], df1, df["position"]], axis=1)

train_set = mmDf[mmDf.position.notna()]
test_set = mmDf[mmDf.position.isna()]

lbEnc = LabelEncoder()
train_set["pos_label"] = lbEnc.fit_transform(train_set.position)

# %%
# feature selection
train_data = train_set.drop(["player","team","season","position"], axis=1)

X = train_data.drop(["pos_label"],axis=1)
y = train_data.pos_label

print(f"X's shape : {X.shape}")
print(f"y's shape : {y.shape}")

xgb = XGBClassifier()
xgb.fit(X, y)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, max_display=40)

#%%
selected_cols = ["height", f"%ast", "oreb%", "dreb%", "%pts_fbps", "%3pm", f"%blk", "ast/to", f"2fgm_%uast",
                 "%pf", "%pts_2pt_mr", "%tov", "stl%", "pitp", "age", "%pfd"]
# selected_cols = ["height", f"%ast", "oreb%", "dreb%", "%pts_fbps", "%3pm", f"%blk", "ast/to", f"2fgm_%uast",
#                  "%pf", "%pts_2pt_mr", "%tov", "stl%", "pitp", "age", "%pfd", "%pts_ft", "ft%", "pf", f"3fgm_%uast",
#                  "poss", "%pts", "to_ratio", "fbps"]

X_sel = X[selected_cols]
y = y

X_train, X_val, y_train, y_val = train_test_split(X_sel, y, test_size=0.2, random_state=64)
print("X_train's shape :", X_train.shape)
print("y_train's shape :", y_train.shape)
print("X_val's shape :", X_val.shape)
print("y_val's shape :", y_val.shape)


# %%
# 포지션 분류 모델 설계

def posClassifierModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, EPOCHS, opti, lossFunc, NORM, NUM_CLASSES, BATCH_SIZE, INITIALIZER):
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS, input_dim=INPUT_DIM, activation='relu', kernel_initializer=INITIALIZER))
    model.add(Dropout(0.1))
    model.add(Dense(HIDDEN_UNITS, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.summary()
    model.compile(optimizer=opti, loss=lossFunc, metrics=['accuracy'])
    model.fit(X_train, y_train,
            validation_data=(X_val, y_val), 
            epochs=EPOCHS, 
            verbose=1,
            batch_size=BATCH_SIZE)
    
    return model

# %%
HIDDEN_UNITS = 128
EPOCHS = 500
BATCH_SIZE = 32
opti = Nadam(learning_rate=0.005)
lossFunc = SparseCategoricalCrossentropy()
NORM = regularizers.l2(0.01)
NUM_CLASSES = 5
INPUT_DIM = 16
INITIALIZER = HeNormal()

model = posClassifierModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, EPOCHS, opti, lossFunc, NORM, NUM_CLASSES, BATCH_SIZE, INITIALIZER)

# %%
# 모델 평가
model.evaluate(X_val, y_val, verbose=1)

# %%
# 포지션 예측값 채우기
X_test = test_set[selected_cols]

preds = model.predict(X_test)

predictions = []
for pred in preds:
    predictions.append(np.argmax(pred))

y_pred = np.array(predictions).reshape(-1,)
y_pred1 = lbEnc.inverse_transform(y_pred)
print(y_pred1)

test_set.position = y_pred1

# %%
resultDf = pd.concat([train_set, test_set], axis=0).drop(["pos_label"], axis=1)
resultDf = resultDf.sort_index()
resultDf

# %%
resultDf.to_csv("./data/filledPosition.csv")

# %%
test_set
# %%
