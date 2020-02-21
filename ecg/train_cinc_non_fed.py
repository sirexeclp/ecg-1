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
# %load_ext autoreload
# %autoreload 2
import tensorflow.keras
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import sys

os.chdir("/workspace/telemed5000/ecg/ecg")
import network
import load
from MITDBDataProvider import *
os.chdir('../')

# %%
params = json.load(open("examples/cinc17/config.json", 'r'))
params.update({
    "input_shape": [None, 1],
    "num_categories": 4
})

MAX_EPOCHS = 100

def create_model():
    with tf.device('/gpu:2'):
        model = network.build_network(**params)
        optimizer = Adam(
            lr=params["learning_rate"],
            clipnorm=params.get("clipnorm", 1))

        model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
        return model

model = create_model()

# %%
with tf.device('/gpu:2'):
    print(f"Loading training set...{params['train']}")
    train = load.load_dataset(params['train'])
    print("Loading dev set...")
    dev = load.load_dataset(params['dev'])
    print("Building preprocessor...")
    preproc = load.Preproc(*train)
    print("Training size: " + str(len(train[0])) + " examples.")
    print("Dev size: " + str(len(dev[0])) + " examples.")

    batch_size = params.get("batch_size", 32)

    stopping = tensorflow.keras.callbacks.EarlyStopping(patience=8)

    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=2,
        min_lr=params["learning_rate"] * 0.001)

    train_x, train_y = preproc.process(*train)
    dev_x, dev_y = preproc.process(*dev)

# %%
with tf.device('/gpu:2'):
    history = model.fit(train_x, train_y,
        batch_size=batch_size,
        epochs=MAX_EPOCHS,
        validation_data=(dev_x, dev_y),
        callbacks=[reduce_lr, stopping])

# %%
model.save("reference_cinc_non_fed.h5")

# %%
import json
import pickle
#with open("train_cinc_non_fed_hist.pkl", "wb") as f:
#    pickle.dump(history.history, f)

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# %%
hist = history.history
def make_nice_plot(hist, metric, annotation=""):
    val_metric = "val_"+metric
    plt.plot(hist[metric])
    plt.plot(hist[val_metric])
    plt.legend([metric, val_metric])
    plt.title(f"{metric.capitalize()} vs. Epochs {annotation}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.savefig(f"{metric.capitalize()} vs. Epochs {annotation}.png")


# %%
make_nice_plot(hist,"loss", "(cinc17 non federated)")

# %%
make_nice_plot(hist, "acc", "(cinc17 non federated)")

# %%

    

