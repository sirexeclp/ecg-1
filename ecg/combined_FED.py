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
import sys
gpu = sys.argv[-1]
gpu = "1"


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
    with tf.device('/gpu:'+gpu):
        model = network.build_network(**params)
        optimizer = Adam(
            lr=params["learning_rate"],
            clipnorm=params.get("clipnorm", 1))

        model.compile(loss='categorical_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
        return model

def fit_model(model, index, dataLoader):
    with tf.device('/gpu:'+gpu):
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
    
    history = []
    for epoch in range(federated_epochs):
        print(f"Epoch {epoch}/{federated_epochs}")
        for index, model in enumerate(models):
            fit_model(model, index, loader)
        new_weights = average_weights(models)
        for model in models:
            model.set_weights(new_weights)
        score = model.evaluate(loader.validationData[0], loader.validationData[1], verbose=0)
        history.append(score)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    return model, history


# %%

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
        reccords_with_afib_and_sinus = ["201", "202",
                                        "203", "219", "222"]
        x_train, x_test, y_train, y_test = get_data_4_fed(file_path
                                   ,record_list=reccords_with_afib_and_sinus)
        x_train, y_train = zip(*[do_oversampling(np.array(x),np.array(y))
                            for x,y in zip(x_train, y_train)])
        
        self.x_train = self.prepare_x_train(x_train)
        self.x_test = self.prepare_x(x_test)
        self.y_train = self.prepare_y_train(y_train)
        self.y_test = self.prepare_y(y_test)
        self.validationData = [np.expand_dims(np.array(self.x_test), axis=2), np.array(self.y_test)]
    
    def prepare_y(self, data_chunk):
        return [np.tile(y_1, (168, 1)) for y_1 in data_chunk]

    def prepare_y_train(self, data):
        return [self.prepare_y(y_1) for y_1 in data]
    
    def prepare_x(self, data_chunk):
        return [x_1[:len(x_1)-192] for x_1 in data_chunk]
    
    def prepare_x_train(self, data):
        return [self.prepare_x(x_1) for x_1 in data]
    
    #def clusterPatients(self):
        
        
    def clientData(self, cluster):
        return [np.expand_dims(np.array(self.x_train[cluster]),axis=2), np.array(self.y_train[cluster])]


# %%
class CombinedFederatedDataLoader:
    
    def __init__(self, mit_file_path, cinc_train_path, cinc_validation_path, clients_count):
        self.mitLoader = MITFederatedDataLoader(file_path)
        self.cincLoader = CINCFederatedDataLoader(cinc_train_path, cinc_validation_path, clients_count)
        
    def clientData(self, cluster):
        print(cluster)
        if cluster <= 4:
            return self.mitLoader.clientData(cluster)
        
        return self.cincLoader.clientData(4 - cluster)


# %%
def do_oversampling(x, y):
    classes = np.argmax(y,axis=1)
    class_counts=np.bincount(classes)
    max_class = np.max(class_counts)
    oversample = max_class - class_counts
    
    result_y = y
    result_x = x
    idx = [np.random.choice(np.where(classes == c)[0],
                            oversample[c]) for c,c_count in enumerate(class_counts)
                           if c_count > 0 ]
    for i in idx:
        result_y = np.concatenate([result_y, y[i]])
        result_x = np.concatenate([result_x, x[i]])
    
    classes = np.argmax(result_y,axis=1)
    class_counts=np.bincount(classes)
    print(class_counts)
    return result_x, result_y


# %% [markdown]
# # Fit cinc17

# %%
file_path = "/workspace/telemed5000/code/data/"
data_json = "examples/cinc17/train.json"
clients_count = 10
epochs = 30
combinedLoader = CombinedFederatedDataLoader(file_path, data_json, params['dev'], clients_count)
model, history = fit_federated(clients_count=clients_count, federated_epochs=epochs,loader=combinedLoader)

# %%
import pickle
with open(f"combined_mit_fed_{clients_count}_{epochs}.pkl", "wb") as f:
    pickle.dump(history, f)

# %%
model.save(f"combined_mit_fed_{clients_count}_{epochs}.h5")

# %%
combinedLoader.clientData(1)

# %%
