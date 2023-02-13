#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier


from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Model, Sequential, regularizers
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from keras.losses import SparseCategoricalCrossentropy
from keras.initializers.initializers_v1 import HeNormal
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


import shap

#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
import tensorflow as tf
tf.test.is_gpu_available()

#%%
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%
df = pd.read_csv("./nonCorr1.csv").drop(["Unnamed: 0"], axis=1)
df

#%%
df1 = df.drop(["player", "team", "season", "position", "inflation_salary"], axis=1)

mmSc = MinMaxScaler()
df2 = pd.DataFrame(mmSc.fit_transform(df1), columns=df1.columns)
mmDf = pd.concat([df[["player", "team", "season"]], df2, df["position"]], axis=1)

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
selected_cols = ["height", "oreb%", f"%ast", "dreb%", "%pts_fbps", f"%blk", "%3pm", f"2fgm_%uast", "%pf", 
                 "ast/to", "%pts_2pt_mr", "stl%", "%pfd", "age", "pitp", f"3fgm_%uast"]
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

def posClassifierModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, EPOCHS, opti, lossFunc, NORM, NUM_CLASSES, BATCH_SIZE, INITIALIZER, checkpoint):
    model = Sequential()
    model.add(Dense(HIDDEN_UNITS, input_dim=INPUT_DIM, activation='relu', kernel_initializer=INITIALIZER))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu', kernel_regularizer=NORM))
    model.add(Dropout(0.1))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.summary()
    model.compile(optimizer=opti, loss=lossFunc, metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val), 
                        epochs=EPOCHS, 
                        verbose=1,
                        batch_size=BATCH_SIZE,
                        callbacks=[checkpoint])
    
    return model, history

# %%
HIDDEN_UNITS = 128
EPOCHS = 1000
BATCH_SIZE = 32
opti = Nadam(learning_rate=0.0005)
lossFunc = SparseCategoricalCrossentropy()
NORM = regularizers.l2(0.1)
NUM_CLASSES = 5
INPUT_DIM = 16
INITIALIZER = HeNormal()

save_path = './model_save/'+'{epoch:03d}-{val_accuracy:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

model, history = posClassifierModel(X_train, y_train, X_val, y_val, HIDDEN_UNITS, INPUT_DIM, 
                                    EPOCHS, opti, lossFunc, NORM, NUM_CLASSES, BATCH_SIZE, 
                                    INITIALIZER, checkpoint)


#%%
# 모델 평가
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
hdf5_path = './model_save/temp/521-0.8012.hdf5'
loaded_model = load_model(hdf5_path, custom_objects={'rmse': rmse})

loaded_model.evaluate(X_val, y_val, verbose=1)

# %%
# 포지션 예측값 채우기
X_test = test_set[selected_cols]

preds = loaded_model.predict(X_test)

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
df.position = resultDf.position
df

# %%
df.to_csv("./data/nonCorrAllPos.csv")

# %%
df.position.isna().sum()
# %%
