# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys,os
sys.path.append("./")

# %%
#import keras
import os
import predict
import util
import load
import scipy.signal as sig
import tqdm

# %%
data_json, model_path = "../examples/cinc17/dev.json","../saved/cinc17/1575022999-481/0.384-0.890-019-0.301-0.898.hdf5"

# %%
predidctions = predict.predict(data_json,model_path)

# %%
preproc = util.load(os.path.dirname(model_path))
dataset = load.load_dataset(data_json)
x, y = preproc.process(*dataset)
x.shape

# %%
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
#conf = confusion_matrix(np.argmax(y,axis=1), np.argmax(predidctions,axis=1))

# %%
predidctions.shape

# %%
pred_reshaped = np.reshape(predidctions,(predidctions.shape[0]*predidctions.shape[1],predidctions.shape[2]))

# %%
y_reshaped = np.reshape(y,(y.shape[0]*y.shape[1],y.shape[2]))

# %%
conf = confusion_matrix(np.argmax(y_reshaped,axis=1), np.argmax(pred_reshaped,axis=1))
conf

# %%
import seaborn as sns

# %%
sns.heatmap(conf, annot=True, fmt="d")

# %%
import json
print(classification_report(np.argmax(y_reshaped,axis=1), np.argmax(pred_reshaped,axis=1),target_names=preproc.classes))

# %%
preproc.classes

# %%
x_hpi = np.load("../examples/hpi/X.npy")
y_hpi = np.load("../examples/hpi/Y.npy")

# %%
x_hpi_resampled = []
for trace in tqdm.tqdm(x_hpi):
    tmp = sig.resample(trace, len(trace) // 360 * 200)
    x_hpi_resampled.append(tmp[:18176])
x_hpi_resampled = np.array(x_hpi_resampled)
x_hpi_resampled = np.expand_dims(x_hpi_resampled, axis=2)

# %%
x_hpi_resampled.shape

# %%
x.shape

# %%
y.shape

# %%
24000/200

# %%
18176/200

# %%
from keras import backend as K
K.clear_session()

# %%
from numba import cuda
cuda.select_device(0)
cuda.close()

# %%
model = keras.models.load_model(model_path)

# %%
from tensorflow.keras.utils import plot_model
import pydot
#plot_model(model)

# %%
predictions = model.predict(x_hpi_resampled)

# %%
conf = confusion_matrix(np.argmax(y_hpi,axis=1), np.argmax(np.mean(predictions, axis=1),axis=1))

# %%
sns.heatmap(conf, annot=True, fmt="d")

# %%
np.argnp.mean(predictions, axis=1).shape

# %%
np.argmax(np.mean(predictions, axis=1),axis=1)

# %%
import matplotlib.pyplot as plt
plt.plot(x_hpi_resampled[0][0:500])

# %%
