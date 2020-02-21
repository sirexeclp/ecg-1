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


# %% [markdown]
# # The cool stuff

# %%
def average_weights(models):
    weights = [model.get_weights() for model in models]
    new_weights = list()

    for weights_list_tuple in zip(*weights):
        new_weights.append(
            np.array([np.array(weights_).mean(axis=0)\
                for weights_ in zip(*weights_list_tuple)]))
    return new_weights


# %%
params = json.load(open("examples/cinc17/config.json", 'r'))
params.update({
    "input_shape": [None, 1],
    "num_categories": 4
})
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

def fit_model(model, index, dataLoader):
    with tf.device('/gpu:2'):
        train_x, train_y = dataLoader.clientData(index)
        model.fit(train_x, train_y)
        return model


# %%
def fit_federated(clients_count, federated_epochs, loader):
    models = list()
    print("Starting to create models...")
    for i in range(clients_count):
        sys.stdout.write("\r\x1b[K"+(i+1).__str__())
        sys.stdout.flush()
        model = create_model()
        models.append(model)
        
    print("Starting the training...")

    for epoch in range(federated_epochs):
        for index, model in enumerate(models):
            fit_model(model, index, loader)
        new_weights = average_weights(models)
        for model in models:
            model.set_weights(new_weights)
        score = model.evaluate(loader.validationData[0], loader.validationData[1], verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    return model


# %% [markdown]
# # Data Loader

# %%
class CINCFederatedDataLoader:
    
    def __init__(self, train_path, validation_path, clients):
        plain_data = load.load_dataset(train_path)
        self.allData = load.Preproc(*plain_data).process(*plain_data)
        self.clients = clients
        self.groupClientData()
        plain_validation_data = load.load_dataset(validation_path)
        preproc = load.Preproc(*plain_validation_data)
        self.validationData = preproc.process(*plain_validation_data)
    
    def clientData(self, index):
        return self.clientsData[index]
    
    def groupClientData(self):
        self.clientsData = []
        xSplit = self.splitArray(self.allData[0])
        ySplit = self.splitArray(self.allData[1])
        for index in range(self.clients):
            self.clientsData.append([xSplit[index], ySplit[index]])

    def splitArray(self, array):
        return np.array_split(array, self.clients)


# %%
class MITFederatedDataLoader:
    
    def __init__(self, file_path):
        # x train x test y train y test
        rawData = get_data_4_fed(file_path)
        self.x_train = self.prepare_x_train(rawData[0])
        self.x_test = self.prepare_x(rawData[1])
        self.y_train = self.prepare_y_train(rawData[2])
        self.y_test = self.prepare_y(rawData[3])
        self.validationData = [np.expand_dims(np.array(self.x_test), axis=2), np.array(self.y_test)]
    def prepare_y(self, data_chunk):
        return [np.tile(y_1, (168, 1)) for y_1 in data_chunk]

    def prepare_y_train(self, data):
        return [self.prepare_y(y_1) for y_1 in data]
    
    def prepare_x(self, data_chunk):
        return [x_1[:len(x_1)-192] for x_1 in data_chunk]
    
    def prepare_x_train(self, data):
        return [self.prepare_x(x_1) for x_1 in data]
        
    def clientData(self, index):
        return [np.expand_dims(np.array(self.x_train[index]),axis=2), np.array(self.y_train[index])]


# %% [markdown]
# # Fit cinc17

# %%
data_json = "examples/cinc17/train.json"
cincLoader = CINCFederatedDataLoader(data_json, params['dev'], 5)
#cincModel = fit_federated(5, 1, cincLoader)
cincLoader.clientData(1)[0].shape

# %% [markdown]
# # Fit MITDB

# %%
file_path = "/workspace/telemed5000/code/data/"
loader = MITFederatedDataLoader(file_path)

mITModel = fit_federated(47, 100, loader)
#mITModel = fit_federated(1, 1, loader)

# %%
tmp_val = [np.expand_dims(loader.validationData[0], axis=2), loader.validationData[1]]


# %%
# loader = MITFederatedDataLoader(file_path)

score = mITModel.evaluate(loader.validationData[0], loader.validationData[1], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#mITModel

# %%
tmp_val

# %%
loader.prepare_y(loader.y)

# %%
y2 = np.load('ecg/MIT/' + 'Y_MIT.npy')

# %%
np.squeeze(y2)

# %%
y = np.array()
for _y in y2:
    _y = np.append(_y, [0, 0])
    print(_y)
    y = np.concatenate(y, _y)

# %%

# %%
